import cv2
import re
import numpy as np
import easyocr
import torch
import time
import threading


class ScreenAnalyzer:
    # How often to grab + OCR the screen. OCR is the most expensive part of the
    # whole app, so this is deliberately coarse. Exposed as a constant so it is
    # easy to tune for slower/faster machines.
    SCREEN_INTERVAL_SEC = 8
    ERROR_BACKOFF_SEC = 10

    def __init__(self, label_provider=None, enabled_provider=None):
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.gpu_flag = True
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_flag = True
        else:
            self.device = "cpu"
            self.gpu_flag = False

        # easyocr only has solid GPU support on CUDA; mps falls back badly, so
        # only hand it the GPU when we actually have CUDA.
        self.reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))

        # The zero-shot classifier is only the *fallback* path (most screens are
        # decided by keywords/heuristics first), and it's the single most
        # expensive thing to load (~5s). So it is built lazily on first use,
        # behind a lock, which keeps initial startup fast. warmup() pre-builds
        # it on a background thread so the first real fallback isn't slow either.
        self._classifier = None
        self._classifier_lock = threading.Lock()

        # Clearer, more separable hypotheses help the zero-shot model commit
        # to a class instead of hedging toward neutral.
        self.LABEL_PRODUCTIVE = "programming, studying, or schoolwork"
        self.LABEL_DISTRACTED = "entertainment, social media, gaming, or shopping"
        self.LABEL_NEUTRAL = "an empty desktop or idle screen"

        # Hypotheses used specifically to judge a single YouTube video by title.
        self.YT_EDU = "an educational tutorial, lecture, course, or how-to guide"
        self.YT_ENT = "entertainment such as gaming, music, comedy, vlogs, or reactions"
        self.MACRO_LABELS = [self.LABEL_PRODUCTIVE, self.LABEL_DISTRACTED, self.LABEL_NEUTRAL]
        self.label_provider = label_provider
        self.enabled_provider = enabled_provider
        self.state_history = []
        self.history_len = 3
        self.coding_markers = [
            r"def\s+\w+\(", r"class\s+\w+", r"import\s+\w+", r"from\s+\w+\s+import",
            r"const\s+\w+\s*=", r"let\s+\w+\s*=", r"function\s+\w+\(", r"fn\s+\w+\(",
            r"#include\s+<\w+>", r"public\s+class\s+\w+", r"console\.log\(", r"printf\("
        ]
        self.strong_productive_keywords = {
            "def ", "import ", "const ", "let ", "var ", "console.log", "public class",
            "fn ", "impl ", "std::", "iostream", "#include", "printf(", "scanf(",
            "using namespace", "public static void main", "System.out.println",
            "Console.WriteLine", "try {", "catch (", "finally {", "async function",
            "await ", "yield ", "lambda ", "struct ", "interface ", "enum ",
            "extends ", "implements ", "export default", "module.exports",

            "select * from", "insert into", "update set", "delete from", "join ",
            "inner join", "group by", "order by", "create table", "drop table",
            "alter table", "document.get", "db.collection", "mongoose.connect",

            "git commit", "git push", "git checkout", "git clone", "git status",
            "git pull", "git merge", "git rebase", "git branch", "git log",
            "npm run", "npm install", "npm start", "npm test", "yarn add", "yarn start",
            "pip install", "pip freeze", "conda activate", "python -m", "cargo build",
            "cargo run", "make build", "gcc ", "g++ ", "cmake ", "mvn clean", "gradlew",
            "docker run", "docker build", "docker-compose up", "kubectl ", "terraform ",

            "package.json", "cargo.toml", "dockerfile", "docker-compose.yml",
            "requirements.txt", "pom.xml", "build.gradle", "tsconfig.json",
            ".gitignore", "webpack.config",

            "localhost:", "127.0.0.1:", "8080", "3000", "5000", "stack overflow",
            "stackoverflow.com", "github.com", "gitlab.com", "bitbucket.org",
            "leetcode.com", "hackerrank.com", "geeksforgeeks.org", "mdn web docs",
            "developer.mozilla.org", "w3schools.com", "freecodecamp.org", "codecademy.com",
            "kaggle.com", "huggingface.co", "vs code", "visual studio code", "pycharm",
            "intellij", "webstorm", "cursor editor", "jupyter notebook", "google colab",
            "copilot", "chatgpt", "claude.ai", "gemini.google", "postman", "insomnia",

            "calculator", "desmos.com", "wolframalpha.com", "wolfram alpha",
            "geogebra.org", "symbolab.com", "mathway.com", "overleaf.com", "arxiv.org",
            "trello.com", "jira.com", "figma.com", "autocad", "solidworks", "lucidchart.com",

            "quizlet.com", "khan academy", "coursera.org", "blackboard.com",
            "canvas.instructure", "edx.org", "udemy.com", "pluralsight.com",
            "datacamp.com", "skillshare.com", "google scholar", "researchgate.net",
            "moodle", "turnitin"
        }

        self.ignore_terms = {
            "mindlens", "live focus workspace", "focus score", "saved sessions",
            "camera feed", "session analytics", "previous sessions", "custom labels",
            "camera detection", "screen detection", "productive", "distracted",
            "neutral", "total session", "override detection",

            "dashboard", "settings", "preferences", "user profile", "account info",
            "sign in", "sign out", "login", "logout", "billing", "subscription",
            "upgrade to pro", "help center", "customer support", "terms of service",
            "privacy policy", "notifications", "app.mindlens", "mindlens.com",
            "focus mode active", "distraction blocked", "tracker daemon", "agent status"
        }

        self.productive_terms = {
            "python", "javascript", "typescript", "java", "c++", "c#", "rust",
            "golang", "ruby", "php", "swift", "kotlin", "scala", "dart", "react",
            "angular", "vue", "svelte", "next.js", "flask", "django", "spring boot",
            "laravel", "ruby on rails", "react native", "flutter",

            "numpy", "pandas", "opencv", "mediapipe", "transformers", "pytorch",
            "tensorflow", "scikit-learn", "api", "function", "class", "import", "def ",
            "const ", "return", "git", "docker", "kubernetes", "aws", "azure", "gcp",
            "sql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "linux",
            "ubuntu", "terminal", "powershell", "bash", "vim", "neovim", "documentation", "docs",

            "github", "stackoverflow", "stack overflow", "visual studio code", "vs code",
            "cursor", "pycharm", "sublime", "intellij", "eclipse", "xcode", "webstorm",
            "postman", "wireshark", "developer", "programming", "coding", "compiler", "debugger",

            "calculator", "calc", "math", "algebra", "geometry", "calculus", "desmos",
            "wolfram", "statistics", "equation", "theorem", "formula", "physics",
            "chemistry", "biology", "economics", "finance", "accounting", "anatomy",
            "thermodynamics", "mechanics", "matrix", "integral", "derivative", "graphing",
            "machine learning", "artificial intelligence", "neural network", "deep learning",

            "figma", "adobe xd", "sketch", "photoshop", "illustrator", "premiere pro",
            "after effects", "blender", "unity", "unreal engine", "wireframe", "ui/ux",

            "notion", "canvas", "blackboard", "quizlet", "khan academy", "coursera",
            "leetcode", "overleaf", "latex", "paper", "research", "assignment",
            "homework", "lecture", "syllabus", "textbook", "study guide", "essay",
            "thesis", "dissertation", "presentation", "report", "proposal",

            "excel", "sheets", "powerpoint", "slides", "word", "pages", "numbers",
            "slack", "teams", "jira", "trello", "asana", "monday.com", "zoom",
            "meet", "meeting", "calendar", "notes", "pdf", "preview", "drive",
            "dropbox", "evernote", "obsidian", "onenote", "resume", "cv", "cover letter",
            "email", "gmail", "outlook", "inbox", "schedule", "planner", "todo",

            # UI-chrome / context words that OCR reliably reads on work pages.
            "quiz", "exam", "rubric", "grade", "module", "worksheet",
            "flashcards", "chapter", "compile", "commit", "repository", "repo",
            "spreadsheet", "citation", "bibliography", "scholar", "merge",
            "due date", "problem set", "lecture notes", "pull request",
            "merge request", "stack trace", "pivot table", "submit assignment",
            "study session", "office hours", "lab report"
        }

        self.distracted_terms = {
            "youtube", "netflix", "hulu", "disney+", "prime video", "hbo max",
            "peacock", "paramount+", "apple tv", "crunchyroll", "funimation",
            "vudu", "tubi", "vimeo", "dailymotion", "twitch",

            "tiktok", "instagram", "reddit", "twitter", "x.com", "facebook",
            "snapchat", "pinterest", "tumblr", "bereal", "threads", "bluesky",
            "mastodon", "weibo", "discord", "whatsapp", "telegram", "wechat",

            "game", "gaming", "roblox", "minecraft", "fortnite", "steam",
            "epic games", "battle.net", "origin", "ubisoft", "league of legends",
            "valorant", "cs:go", "cs2", "dota 2", "apex legends", "call of duty",
            "warzone", "overwatch", "genshin impact", "honkai", "xbox",
            "playstation", "nintendo", "ign", "kotaku", "polygon",

            "memes", "shorts", "reels", "buzzfeed", "imgur", "9gag", "4chan",

            "spotify", "apple music", "soundcloud", "pandora", "tidal", "last.fm",

            "amazon.com", "ebay", "walmart", "target", "etsy", "aliexpress",
            "temu", "shein", "craigslist", "zillow", "wayfair", "asos",

            "tmz", "daily mail", "fox news", "cnn", "nytimes", "huffpost",
            "espn", "bleacher report", "fantasy football", "sports betting", "draftkings",

            "tinder", "bumble", "hinge", "okcupid", "match.com",

            # UI-chrome / context words that OCR reliably reads on these pages.
            "subscribe", "subscribers", "trending", "followers", "retweet",
            "upvote", "downvote", "matchmaking", "respawn", "leaderboard",
            "episode", "episodes", "checkout", "wishlist",
            "watch later", "up next", "recommended for you", "live chat",
            "now playing", "continue watching", "my list", "add to cart",
            "buy now", "free shipping", "add to bag", "battle pass",
            "trending now", "for you page"
        }

        # ── Decisive site / app identity ────────────────────────────────────
        # The single most reliable signal of what you are *actually* doing is
        # which site or app is open — not stray keywords elsewhere on screen.
        # These are checked first and dominate the decision, so a focus-sounding
        # video title (or a view count that happens to read like a port number)
        # can never make YouTube count as productive.
        self.distracting_sites = {
            "netflix", "hulu", "disney+", "disneyplus",
            "hbo max", "max.com", "prime video", "primevideo", "peacock",
            "paramount+", "crunchyroll", "funimation", "twitch", "tubi.tv",
            "tiktok", "instagram", "reddit", "facebook", "snapchat", "pinterest",
            "tumblr", "threads.net", "bluesky", "mastodon", "twitter", "x.com",
            "9gag", "imgur", "4chan", "buzzfeed", "fandom",
            "spotify", "soundcloud", "pandora", "tidal",
            "roblox", "minecraft", "fortnite", "steampowered", "steamcommunity",
            "epic games", "battle.net", "valorant", "league of legends",
            "genshin", "honkai", "call of duty", "warzone", "overwatch",
            "apex legends", "ign.com", "kotaku", "polygon.com",
            "amazon.com", "ebay.com", "aliexpress", "temu.com", "shein",
            "etsy.com", "walmart.com", "target.com", "wayfair", "asos",
            "tinder", "bumble", "hinge", "onlyfans",
            "espn.com", "draftkings", "fanduel", "bleacher report",
        }
        self.productive_sites = {
            "github", "gitlab", "bitbucket", "stack overflow", "stackoverflow",
            "leetcode", "hackerrank", "codeforces", "geeksforgeeks", "codecademy",
            "freecodecamp", "kaggle", "huggingface", "replit", "codepen",
            "jsfiddle", "codesandbox", "mdn web docs", "developer.mozilla",
            "w3schools", "readthedocs", "devdocs",
            "visual studio code", "vs code", "pycharm", "intellij", "webstorm",
            "android studio", "xcode", "jupyter", "google colab", "colab.research",
            "overleaf", "notion.so", "obsidian", "onenote",
            "google docs", "docs.google", "sheets.google", "slides.google",
            "google scholar", "scholar.google", "arxiv", "researchgate",
            "sciencedirect", "jstor", "pubmed", "ieee",
            "desmos", "wolframalpha", "wolfram alpha", "geogebra", "symbolab",
            "khan academy", "khanacademy", "coursera", "edx.org", "udemy",
            "quizlet", "brilliant.org", "canvas", "blackboard", "moodle",
            "gradescope", "piazza", "instructure", "chegg", "wikipedia",
            "chatgpt", "claude.ai", "gemini.google", "copilot", "perplexity",
            "jira", "trello", "asana", "linear.app", "confluence", "figma",
            "miro.com", "lucidchart", "autocad", "solidworks",
        }
        # Real coding / CLI activity: text that simply does not appear while you
        # are watching a video or scrolling a feed. Decisive for "Productive".
        self.dev_activity = {
            "def ", "import ", "console.log", "printf(", "println", "#include",
            "std::", "public static void", "void main", "system.out", "fn main",
            "git commit", "git push", "git status", "git checkout", "git pull",
            "git merge", "git rebase", "git clone", "npm run", "npm install",
            "npm start", "pip install", "python -m", "cargo build", "cargo run",
            "docker build", "docker run", "kubectl ", "select * from",
            "insert into", "create table", "stack trace", "syntaxerror",
            "nameerror", "typeerror:", "segmentation fault", "npm err",
        }

        # ── YouTube: classify the *video*, not the site ─────────────────────
        # YouTube hosts both lectures and entertainment, so it gets its own
        # content-based path driven mainly by the video title (which YouTube
        # mirrors into the browser tab). These signal lists handle the
        # high-confidence cases; anything ambiguous falls through to a
        # zero-shot judgement of the title.
        self.youtube_tokens = {"youtube", "youtu.be", " - youtube"}
        self.yt_educational = {
            "tutorial", "how to", "how i", "explained", "explainer", "lecture",
            "course", "crash course", "full course", "masterclass", "bootcamp",
            "study", "study with me", "exam", "revision", "walkthrough", "lesson",
            "learn ", "learning", "derivation", "proof", "theorem", "solved",
            "solution", "step by step", "beginners", "for beginners", "fundamentals",
            "introduction to", "intro to", "deep dive", "in depth", "guide to",
            "programming", "coding", "code with", "python", "javascript", "java ",
            "c++", "rust", "golang", "leetcode", "data structures", "algorithm",
            "algorithms", "system design", "interview prep", "calculus", "algebra",
            "geometry", "trigonometry", "statistics", "physics", "chemistry",
            "biology", "anatomy", "economics", "accounting", "finance basics",
            "essay", "thesis", "dissertation", "academic", "professor", "university",
            "lecture notes", "documentation", "docs", "explain", "understanding",
            "ted talk", "tedx", "conference talk", "keynote", "webinar", "seminar",
            "sat prep", "mcat", "ap ", "gcse", "a level", "textbook", "khan academy",
            "neural network", "machine learning", "deep learning", "data science",
            "excel tutorial", "powerpoint", "research paper",
        }
        self.yt_entertainment = {
            "official video", "official music video", "music video", "lyrics",
            "ft.", "feat.", "audio)", "new song", "album", "concert", "live performance",
            "vlog", "vlogs", "prank", "reaction", "reacts", "react ", "funny", "fails",
            "compilation", "meme", "memes", "gameplay", "playthrough", "let's play",
            "lets play", "speedrun", "highlights", "trailer", "teaser", "unboxing",
            "haul", "mukbang", "asmr", "challenge", "tier list", "trolling", "montage",
            "stand up", "stand-up", "comedy", "sketch", "shorts", "best moments",
            "top 10", "top 5", "ranking", "i spent", "24 hours", "100 hours",
            "survived", "craziest", "gone wrong", "story time",
            "storytime", "drama", "exposed", "beef", "tiktok compilation",
            "satisfying", "i bought", "we tried",
            "minecraft", "fortnite", "gta", "roblox", "valorant", "warzone", "fifa",
            "full match", "edit)", "amv",
        }

        # Pre-compile matchers. Plain alphanumeric terms are matched on word
        # boundaries so "ign" (IGN) no longer fires inside "ass-ign-ment" or
        # "des-ign", "api" inside "ther-api-st", or "vim" inside "vim-eo".
        # Terms containing spaces/punctuation/digits keep substring matching.
        self._productive_matcher = self._compile_term_matcher(self.productive_terms)
        self._distracted_matcher = self._compile_term_matcher(self.distracted_terms)

        self.current_state = "Neutral"
        self.running = False
        self.thread = None

    def _compile_term_matcher(self, terms):
        word_terms = []
        substr_terms = []
        for t in terms:
            # Only pure letter tokens are boundary-matched. Anything with a
            # space, dot, digit, +, # etc. (domains, "c++", "stack overflow",
            # "def ") stays a substring test so existing behaviour is preserved.
            if re.fullmatch(r"[a-z]{2,}", t):
                word_terms.append(re.escape(t))
            else:
                substr_terms.append(t)
        pattern = re.compile(r"\b(?:" + "|".join(word_terms) + r")\b") if word_terms else None
        return pattern, substr_terms

    @property
    def classifier(self):
        """Lazily build the zero-shot classifier the first time it's needed."""
        if self._classifier is None:
            with self._classifier_lock:
                if self._classifier is None:
                    from transformers import pipeline
                    self._classifier = pipeline(
                        "zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli",
                        device=self.device,
                    )
        return self._classifier

    def warmup(self):
        """Absorb one-time first-call costs so the first real screen grab is
        fast. OCR is exercised on a tiny dummy image, and the (lazy) classifier
        is built on a background thread since it's the slowest piece and only
        ever runs on the fallback path."""
        threading.Thread(target=lambda: self.classifier, daemon=True).start()
        try:
            dummy = np.zeros((32, 200), dtype=np.uint8)
            self.reader.readtext(dummy, detail=0)
        except Exception as e:
            print(f"[Screen] OCR warmup skipped: {e}")

    def start(self):
        self.running = True
        # Fresh hysteresis history per session so a previous session's tail
        # doesn't bias the first reading of a new one.
        self.state_history = []
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _capture_gray(self):
        """Grab the primary monitor as a grayscale numpy array.

        Uses `mss` (fast, captures a single monitor, and works on Windows,
        macOS and Linux/X11) and falls back to `pyautogui` if mss is missing or
        fails (e.g. some Wayland setups). Capturing only the primary monitor —
        instead of a stitched multi-monitor canvas — keeps the image small and
        OCR fast.
        """
        try:
            import mss  # imported lazily so a missing dep degrades gracefully
            with mss.mss() as sct:
                # monitors[1] is the primary physical monitor; [0] is the union
                # of all monitors, which we deliberately avoid.
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                shot = sct.grab(monitor)
                frame = np.asarray(shot)  # BGRA
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        except Exception:
            import pyautogui
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    def _run_loop(self):
        print("[Screen] Analyzer background thread started...")
        while self.running:
            try:
                if self.enabled_provider and not self.enabled_provider():
                    time.sleep(1)
                    continue

                img = self._capture_gray()
                height, width = img.shape
                max_width = 1600
                if width > max_width:
                    scale = max_width / width
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                raw_results = self.reader.readtext(img, detail=1)

                # Split the OCR into a "header" band (top of the screen: browser
                # tabs, address bar, window title) and the body. The header is
                # where the open site identifies itself, so it's the strongest
                # signal and gets weighted heavily in classification.
                header_cut = img.shape[0] * 0.12
                header_tokens, body_tokens = [], []
                for box, text, conf in raw_results:
                    if conf <= 0.3:
                        continue
                    y_top = min(pt[1] for pt in box)
                    (header_tokens if y_top < header_cut else body_tokens).append(text)

                full_text = " ".join(header_tokens + body_tokens)
                header_text = " ".join(header_tokens)
                self.current_state = self._classify_text(full_text, header_text)

                # Sleep in short slices so stop() / disable is responsive.
                self._interruptible_sleep(self.SCREEN_INTERVAL_SEC)

            except Exception as e:
                print(f"[Screen] Error encountered: {e}")
                self._interruptible_sleep(self.ERROR_BACKOFF_SEC)

    def _interruptible_sleep(self, seconds):
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(0.5, end - time.time()))

    def get_state(self):
        return self.current_state

    def _classify_text(self, full_text, header_text=""):
        raw_lower_text = full_text.lower()
        if self._looks_like_mindlens_dashboard(raw_lower_text):
            raw_state = "Neutral"
        else:
            raw_state = self._get_raw_state(full_text, header_text)

        # Apply state history smoothing (hysteresis)
        self.state_history.append(raw_state)
        if len(self.state_history) > self.history_len:
            self.state_history.pop(0)

        # Return majority vote
        return max(set(self.state_history), key=self.state_history.count)

    def _site_score(self, lower_text, header_lower, sites):
        """Weighted count of site/app identifiers. A match in the header band
        (browser tab / address bar / title) counts double, since that's where
        the *active* site names itself."""
        score = 0
        for s in sites:
            if s in header_lower:
                score += 2
            elif s in lower_text:
                score += 1
        return score

    def _extract_youtube_title(self, header_lower):
        """Pull the video title out of the browser-tab text. YouTube tabs read
        like "(3) How to learn calculus - YouTube - Google Chrome", so we strip
        the notification badge, the "YouTube" suffix and the browser name."""
        t = header_lower
        for junk in (" - youtube", "youtube", "- google chrome", "google chrome",
                     "mozilla firefox", "microsoft edge", "- safari", "safari",
                     "- brave", "brave", "- opera", "arc"):
            t = t.replace(junk, " ")
        t = re.sub(r"\(\d+\)", " ", t)        # "(3)" unread/notification badge
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _classify_youtube(self, header_lower, lower_text):
        """Decide whether the *YouTube video itself* is productive or a
        distraction. The title (mirrored into the browser tab) is the strongest
        signal; we score it against educational vs entertainment vocabulary and,
        when that's inconclusive, let the zero-shot model judge the title."""
        title = self._extract_youtube_title(header_lower)
        # The title carries the most weight; add a little body context for
        # videos where the tab text was missed/garbled by OCR.
        probe = (title + " " + lower_text[:500]).strip()

        edu = sum(1 for t in self.yt_educational if t in probe)
        ent = sum(1 for t in self.yt_entertainment if t in probe)
        # Reinforce with the general study/distraction matchers on the title.
        edu += self._count_hits(title, self._productive_matcher)
        ent += self._count_hits(title, self._distracted_matcher)

        # Confident keyword verdicts.
        if edu and not ent:
            return "Productive"
        if ent and not edu:
            return "Distracted"
        if edu >= ent + 2:
            return "Productive"
        if ent >= edu + 2:
            return "Distracted"

        # Ambiguous (or no keywords): judge the title with the zero-shot model.
        text = (title or lower_text[:300]).strip()
        if len(text) < 4:
            # No readable title — most bare YouTube views are entertainment.
            return "Distracted"
        try:
            result = self.classifier(
                text[:400],
                [self.YT_EDU, self.YT_ENT],
                hypothesis_template="This YouTube video is {}."
            )
            scores = dict(zip(result['labels'], result['scores']))
            edu_s = scores.get(self.YT_EDU, 0.0)
            ent_s = scores.get(self.YT_ENT, 0.0)
        except Exception as e:
            print(f"[Screen] YouTube classify error: {e}")
            return "Distracted"

        # Nudge the model with any (weak) keyword lean before deciding.
        if edu > ent:
            edu_s += 0.10
        elif ent > edu:
            ent_s += 0.10
        return "Productive" if edu_s > ent_s else "Distracted"

    def _get_raw_state(self, full_text, header_text=""):
        cleaned_text = self._clean_text(full_text)
        lower_text = cleaned_text.lower()
        header_lower = (header_text or "").lower()

        if len(lower_text.strip()) < 10:
            return "Neutral"

        # 1. Custom user labels always win. Distracted is checked first so a
        #    blocked site can't be rescued by a productive label appearing
        #    elsewhere on the page.
        productive_labels, distracted_labels = self._custom_labels()
        if self._contains_any(lower_text, distracted_labels):
            return "Distracted"
        if self._contains_any(lower_text, productive_labels):
            return "Productive"

        # 2. Site / app identity dominates. Whatever site is actually open
        #    decides the verdict — stray keywords elsewhere on screen cannot
        #    override it. YouTube is the exception: it's judged by the video.
        d_site = self._site_score(lower_text, header_lower, self.distracting_sites)
        p_site = self._site_score(lower_text, header_lower, self.productive_sites)
        youtube_open = self._contains_any(header_lower, self.youtube_tokens) or (
            "youtube" in lower_text
            and self._contains_any(lower_text,
                                   ("subscribe", "subscribers", "up next",
                                    "views", "watch later", "comments"))
        )

        # A clearly productive site (GitHub, Docs, an LMS…) wins outright.
        if p_site > d_site and p_site > 0:
            return "Productive"
        # A non-YouTube distracting site (Netflix, TikTok, games…) is decisive.
        if d_site > 0 and d_site >= p_site:
            return "Distracted"
        # YouTube: classify the actual video, not the site.
        if youtube_open:
            return self._classify_youtube(header_lower, lower_text)

        # 3. Unambiguous coding / CLI activity. This text never shows up while
        #    watching a video or scrolling a feed, so it's safe to trust.
        if self._contains_any(lower_text, self.dev_activity):
            return "Productive"
        for marker in self.coding_markers:
            if re.search(marker, full_text):
                return "Productive"
        total_chars = len(lower_text.replace(" ", ""))
        if total_chars >= 20:
            special_symbols = sum(1 for c in lower_text if c in "{}[]=;<>")
            if special_symbols / total_chars > 0.08:
                return "Productive"

        # 4. Weighted general keywords (header double-counted). Ties go to
        #    Distracted — the opposite of the old behaviour, which leaked
        #    distractions through as "benefit of the doubt".
        productive_hits = (self._count_hits(lower_text, self._productive_matcher)
                           + self._count_hits(header_lower, self._productive_matcher))
        distracted_hits = (self._count_hits(lower_text, self._distracted_matcher)
                           + self._count_hits(header_lower, self._distracted_matcher))
        if productive_hits or distracted_hits:
            return "Productive" if productive_hits > distracted_hits else "Distracted"

        # 5. Zero-shot classification.
        # The screen has readable text but no decisive keywords, so let the model
        # decide. Thresholds are intentionally low: "Neutral" is reserved for a
        # genuinely blank/idle screen, not just any page the keyword lists missed.
        # This is what previously made everything collapse to Neutral.
        result = self.classifier(
            cleaned_text[:1500],
            self.MACRO_LABELS,
            hypothesis_template="This screen shows {}."
        )
        scores = dict(zip(result['labels'], result['scores']))
        prod = scores.get(self.LABEL_PRODUCTIVE, 0.0)
        dist = scores.get(self.LABEL_DISTRACTED, 0.0)
        neu = scores.get(self.LABEL_NEUTRAL, 0.0)

        text_len = len(lower_text)

        # Only return Neutral when "idle/empty" clearly wins on a sparse screen.
        if neu >= prod and neu >= dist and neu >= 0.50 and text_len < 200:
            return "Neutral"

        # A text-heavy screen almost always means the user is doing *something*;
        # commit to the stronger active class rather than bailing to Neutral.
        decisive = max(prod, dist) >= 0.28 or text_len >= 120
        if not decisive:
            return "Neutral"
        # Ties go to Distracted, consistent with the rest of the pipeline.
        return "Productive" if prod > dist else "Distracted"

    def _clean_text(self, full_text):
        words = full_text.split()
        cleaned_words = []
        for w in words:
            w_low = w.lower()
            if w_low in self.ignore_terms:
                continue
            if len(w) <= 2 and not w.isalnum():
                continue
            cleaned_words.append(w)
        return " ".join(cleaned_words)

    def _looks_like_mindlens_dashboard(self, lower_text):
        dashboard_hits = sum(1 for term in self.ignore_terms if term in lower_text)
        return dashboard_hits >= 3

    def _custom_labels(self):
        if not self.label_provider:
            return [], []
        try:
            productive, distracted = self.label_provider()
            return productive or [], distracted or []
        except Exception:
            return [], []

    def _contains_any(self, lower_text, labels):
        return any(label and label.lower() in lower_text for label in labels)

    def _count_hits(self, lower_text, compiled):
        pattern, substr_terms = compiled
        hits = 0
        if pattern is not None:
            hits += len(set(pattern.findall(lower_text)))
        hits += sum(1 for term in substr_terms if term in lower_text)
        return hits