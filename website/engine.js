/* ============================================================================
   MindLens — in-browser detection engine

   A faithful client-side port of the Python desktop pipeline. Everything runs
   on-device in the browser; no frames ever leave the machine.

     - Camera : getUserMedia -> MediaPipe hand/face landmarkers + TF.js coco-ssd
                object detection -> CameraAnalyzer (port of StateAnalyzer)
     - Screen : getDisplayMedia -> Tesseract.js OCR -> ScreenClassifier
                (port of ScreenAnalyzer's keyword/site decision logic)
     - State  : AppState (port of AppState + the tracker unify/timer loop)

   The engine emits a status object with the exact same shape the dashboard UI
   already expects, so the UI layer (app.js) stays a thin renderer.

   tfjs / coco-ssd / Tesseract are loaded as UMD globals (tf, cocoSsd,
   Tesseract); MediaPipe tasks-vision is imported as an ES module below.
   ========================================================================== */

// MediaPipe tasks-vision is imported dynamically (inside the camera loader) so a
// CDN hiccup can never take down the whole dashboard at module-load time.
const TASKS_VISION_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const HAND_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_BASE =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

/* -------------------------------------------------------------------------- */
/*  Unified state (port of tracker.get_unified_state)                          */
/* -------------------------------------------------------------------------- */
function getUnifiedState(cameraState, screenState) {
    if (cameraState === "Actively Using Phone") return "Distracted";
    if (cameraState === "Studying / Writing" || cameraState === "Reading") return "Productive";
    if (screenState === "Productive") return "Productive";
    if (screenState === "Distracted") return "Distracted";
    return "Neutral";
}

/* ========================================================================== */
/*  CameraAnalyzer — port of CameraDetection/state_analyzer.py                 */
/* ========================================================================== */
class CameraAnalyzer {
    constructor() {
        this.HISTORY_WINDOW_SEC = 1.6;
        this.PHONE_CONFIRM_SEC = 0.5;
        this.history = []; // {t, state}
    }

    _checkOverlap(handCoords, bbox) {
        const [x1, y1, x2, y2] = bbox;
        for (const [hx, hy] of handCoords) {
            if (x1 <= hx && hx <= x2 && y1 <= hy && hy <= y2) return true;
        }
        return false;
    }

    _boxArea(bbox) {
        if (!bbox) return 0;
        const [x1, y1, x2, y2] = bbox;
        return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    }

    _prune(now) {
        const cutoff = now - this.HISTORY_WINDOW_SEC;
        while (this.history.length && this.history[0].t < cutoff) this.history.shift();
    }

    // objects: [{cls:'phone'|'book'|'laptop', bbox:[x1,y1,x2,y2], conf}]
    analyze(objects, handCoords, isPinching, headDown, chinY) {
        let rawState = "Idle";
        let phoneBox = null, bookBox = null, laptopBox = null;
        let phoneConf = 0, bookConf = 0, laptopConf = 0;

        for (const obj of objects) {
            if (obj.cls === "phone" && obj.conf > phoneConf) { phoneConf = obj.conf; phoneBox = obj.bbox; }
            else if (obj.cls === "book" && obj.conf > bookConf) { bookConf = obj.conf; bookBox = obj.bbox; }
            else if (obj.cls === "laptop" && obj.conf > laptopConf) { laptopConf = obj.conf; laptopBox = obj.bbox; }
        }

        let inDeskZone = false;
        for (const [, hy] of handCoords) {
            if (hy > chinY) { inDeskZone = true; break; }
        }

        const hasPhone = phoneBox && this._checkOverlap(handCoords, phoneBox);
        const hasBook = bookBox && this._checkOverlap(handCoords, bookBox);
        const hasLaptop = laptopBox && this._checkOverlap(handCoords, laptopBox);

        const phoneArea = phoneBox ? this._boxArea(phoneBox) : 0;
        const heldPhone = hasPhone && phoneConf >= 0.42;
        const visiblePhone = phoneBox && phoneConf >= 0.48 && phoneArea >= 550;
        const highConfPhone = phoneBox && phoneConf >= 0.62;
        const conflictingStudyObject = (hasLaptop || hasBook) && phoneConf < 0.72;

        if ((heldPhone || visiblePhone || highConfPhone) && !conflictingStudyObject) {
            rawState = "Actively Using Phone";
        } else if (isPinching && inDeskZone) {
            rawState = "Studying / Writing";
        } else if (hasLaptop) {
            rawState = "Studying / Writing";
        } else if (hasBook) {
            rawState = "Reading";
        } else if (headDown && inDeskZone) {
            rawState = "Reading";
        }

        const now = performance.now() / 1000;
        this.history.push({ t: now, state: rawState });
        this._prune(now);

        if (rawState === "Actively Using Phone") {
            const phoneSpan = this.history.filter((h) => h.state === "Actively Using Phone");
            if (phoneSpan.length && now - phoneSpan[0].t >= this.PHONE_CONFIRM_SEC) return "Actively Using Phone";
            if (phoneSpan.length >= 2) return "Actively Using Phone";
        }

        const states = this.history.map((h) => h.state);
        if (!states.length) return rawState;
        const counts = {};
        let best = states[0], bestN = 0;
        for (const s of states) { counts[s] = (counts[s] || 0) + 1; if (counts[s] > bestN) { bestN = counts[s]; best = s; } }
        return best;
    }

    reset() { this.history = []; }
}

/* ========================================================================== */
/*  CameraEngine — getUserMedia + MediaPipe + coco-ssd                         */
/* ========================================================================== */
class CameraEngine {
    constructor(video, overlay) {
        this.video = video;
        this.overlay = overlay;
        this.analyzer = new CameraAnalyzer();
        this.handLandmarker = null;
        this.faceLandmarker = null;
        this.objectModel = null;
        this.stream = null;
        this.running = false;
        this.state = "Idle";
        this.cachedObjects = [];
        this.cachedObjectsTs = 0;
        this.TARGET_AI_FPS = 8;
        this.BOX_STALE_SEC = 0.8;
        this._loadingPromise = null;
        this._lastVideoTs = -1;
    }

    async _loadModels() {
        if (this._loadingPromise) return this._loadingPromise;
        this._loadingPromise = (async () => {
            const { FilesetResolver, HandLandmarker, FaceLandmarker } = await import(TASKS_VISION_URL);
            const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
            [this.handLandmarker, this.faceLandmarker, this.objectModel] = await Promise.all([
                HandLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath: HAND_MODEL_URL },
                    runningMode: "VIDEO",
                    numHands: 2,
                    minHandDetectionConfidence: 0.5,
                    minHandPresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                }),
                FaceLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath: FACE_MODEL_URL },
                    runningMode: "VIDEO",
                    numFaces: 1,
                    minFaceDetectionConfidence: 0.5,
                    minFacePresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                }),
                // eslint-disable-next-line no-undef
                cocoSsd.load({ base: "lite_mobilenet_v2" }),
            ]);
        })();
        return this._loadingPromise;
    }

    async start() {
        await this._loadModels();
        this.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false,
        });
        this.video.srcObject = this.stream;
        await this.video.play();
        this.analyzer.reset();
        this.running = true;
        this.state = "Idle";
        this._aiLoop();
        this._drawLoop();
    }

    stop() {
        this.running = false;
        if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
        this.video.srcObject = null;
        this.cachedObjects = [];
        const ctx = this.overlay.getContext("2d");
        ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        this.state = "Camera Off";
    }

    _detectObjects(predictions) {
        const map = { "cell phone": "phone", book: "book", laptop: "laptop" };
        const out = [];
        for (const p of predictions) {
            const cls = map[p.class];
            if (!cls) continue;
            const [x, y, w, h] = p.bbox;
            out.push({ cls, bbox: [x, y, x + w, y + h], conf: p.score, name: p.class });
        }
        return out;
    }

    _pose(timestamp) {
        const vw = this.video.videoWidth, vh = this.video.videoHeight;
        const handCoords = [];
        let isPinching = false;
        let chinY = vh / 2;
        let headDown = false;

        const handRes = this.handLandmarker.detectForVideo(this.video, timestamp);
        for (const lms of handRes.landmarks || []) {
            for (const lm of lms) handCoords.push([lm.x * vw, lm.y * vh]);
            const wrist = lms[0], knuckle = lms[9];
            let scale = Math.sqrt((wrist.x - knuckle.x) ** 2 + (wrist.y - knuckle.y) ** 2 + (wrist.z - knuckle.z) ** 2) || 1;
            const t = lms[4], i = lms[8], m = lms[12];
            const dTI = Math.sqrt((t.x - i.x) ** 2 + (t.y - i.y) ** 2 + (t.z - i.z) ** 2) / scale;
            const dTM = Math.sqrt((t.x - m.x) ** 2 + (t.y - m.y) ** 2 + (t.z - m.z) ** 2) / scale;
            if (dTI < 0.33 || dTM < 0.33) isPinching = true;
        }

        const faceRes = this.faceLandmarker.detectForVideo(this.video, timestamp);
        if (faceRes.faceLandmarks && faceRes.faceLandmarks.length) {
            const face = faceRes.faceLandmarks[0];
            const forehead = face[10], chin = face[152], nose = face[1];
            chinY = chin.y * vh;
            const faceHeight = Math.abs(forehead.y - chin.y);
            if (faceHeight > 0) {
                const ratio = (chin.y - nose.y) / faceHeight;
                if (ratio < 0.36) headDown = true;
            }
        }
        return { handCoords, isPinching, headDown, chinY };
    }

    async _aiLoop() {
        const minInterval = 1000 / this.TARGET_AI_FPS;
        while (this.running) {
            const loopStart = performance.now();
            if (this.video.readyState >= 2 && this.video.videoWidth) {
                try {
                    // MediaPipe requires a monotonically increasing timestamp.
                    let ts = this.video.currentTime * 1000;
                    if (ts <= this._lastVideoTs) ts = this._lastVideoTs + 1;
                    this._lastVideoTs = ts;

                    const predictions = await this.objectModel.detect(this.video);
                    const objects = this._detectObjects(predictions);
                    const { handCoords, isPinching, headDown, chinY } = this._pose(ts);
                    this.state = this.analyzer.analyze(objects, handCoords, isPinching, headDown, chinY);
                    this.cachedObjects = objects;
                    this.cachedObjectsTs = performance.now();
                } catch (e) {
                    /* transient frame errors are non-fatal */
                }
            }
            const elapsed = performance.now() - loopStart;
            await sleep(Math.max(0, minInterval - elapsed));
        }
    }

    _drawLoop() {
        const ctx = this.overlay.getContext("2d");
        const tick = () => {
            if (!this.running) return;
            const vw = this.video.videoWidth, vh = this.video.videoHeight;
            if (vw && vh) {
                if (this.overlay.width !== vw || this.overlay.height !== vh) {
                    this.overlay.width = vw;
                    this.overlay.height = vh;
                }
                ctx.clearRect(0, 0, vw, vh);
                const fresh = (performance.now() - this.cachedObjectsTs) / 1000 <= this.BOX_STALE_SEC;
                if (fresh) {
                    for (const obj of this.cachedObjects) {
                        const [x1, y1, x2, y2] = obj.bbox;
                        ctx.fillStyle = "rgba(110, 139, 255, 0.12)";
                        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                        ctx.strokeStyle = "rgba(110, 139, 255, 0.95)";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        const label = `${obj.name} ${Math.round(obj.conf * 100)}%`;
                        ctx.font = "600 14px Inter, sans-serif";
                        const tw = ctx.measureText(label).width;
                        ctx.fillStyle = "rgba(110, 139, 255, 0.95)";
                        ctx.fillRect(x1, Math.max(0, y1 - 20), tw + 12, 20);
                        ctx.fillStyle = "#0a0a0c";
                        ctx.fillText(label, x1 + 6, Math.max(14, y1 - 6));
                    }
                }
            }
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }
}

/* ========================================================================== */
/*  ScreenClassifier — port of ScreenDetection/screen_analyzer.py decision     */
/* ========================================================================== */
class ScreenClassifier {
    constructor() {
        this.history_len = 3;
        this.state_history = [];

        this.coding_markers = [
            /def\s+\w+\(/, /class\s+\w+/, /import\s+\w+/, /from\s+\w+\s+import/,
            /const\s+\w+\s*=/, /let\s+\w+\s*=/, /function\s+\w+\(/, /fn\s+\w+\(/,
            /#include\s+<\w+>/, /public\s+class\s+\w+/, /console\.log\(/, /printf\(/,
        ];

        this.ignore_terms = new Set([
            "mindlens", "live focus workspace", "focus score", "saved sessions",
            "camera feed", "session analytics", "previous sessions", "custom labels",
            "camera detection", "screen detection", "productive", "distracted",
            "neutral", "total session", "override detection",
            "dashboard", "settings", "preferences", "user profile", "account info",
            "sign in", "sign out", "login", "logout", "billing", "subscription",
            "upgrade to pro", "help center", "customer support", "terms of service",
            "privacy policy", "notifications", "app.mindlens", "mindlens.com",
            "focus mode active", "distraction blocked", "tracker daemon", "agent status",
        ]);

        this.productive_terms = new Set([
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
            "quiz", "exam", "rubric", "grade", "module", "worksheet",
            "flashcards", "chapter", "compile", "commit", "repository", "repo",
            "spreadsheet", "citation", "bibliography", "scholar", "merge",
            "due date", "problem set", "lecture notes", "pull request",
            "merge request", "stack trace", "pivot table", "submit assignment",
            "study session", "office hours", "lab report",
        ]);

        this.distracted_terms = new Set([
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
            "subscribe", "subscribers", "trending", "followers", "retweet",
            "upvote", "downvote", "matchmaking", "respawn", "leaderboard",
            "episode", "episodes", "checkout", "wishlist",
            "watch later", "up next", "recommended for you", "live chat",
            "now playing", "continue watching", "my list", "add to cart",
            "buy now", "free shipping", "add to bag", "battle pass",
            "trending now", "for you page",
        ]);

        this.distracting_sites = new Set([
            "netflix", "hulu", "disney+", "disneyplus", "hbo max", "max.com",
            "prime video", "primevideo", "peacock", "paramount+", "crunchyroll",
            "funimation", "twitch", "tubi.tv", "tiktok", "instagram", "reddit",
            "facebook", "snapchat", "pinterest", "tumblr", "threads.net", "bluesky",
            "mastodon", "twitter", "x.com", "9gag", "imgur", "4chan", "buzzfeed",
            "fandom", "spotify", "soundcloud", "pandora", "tidal", "roblox",
            "minecraft", "fortnite", "steampowered", "steamcommunity", "epic games",
            "battle.net", "valorant", "league of legends", "genshin", "honkai",
            "call of duty", "warzone", "overwatch", "apex legends", "ign.com",
            "kotaku", "polygon.com", "amazon.com", "ebay.com", "aliexpress",
            "temu.com", "shein", "etsy.com", "walmart.com", "target.com", "wayfair",
            "asos", "tinder", "bumble", "hinge", "onlyfans", "espn.com",
            "draftkings", "fanduel", "bleacher report",
        ]);

        this.productive_sites = new Set([
            "github", "gitlab", "bitbucket", "stack overflow", "stackoverflow",
            "leetcode", "hackerrank", "codeforces", "geeksforgeeks", "codecademy",
            "freecodecamp", "kaggle", "huggingface", "replit", "codepen",
            "jsfiddle", "codesandbox", "mdn web docs", "developer.mozilla",
            "w3schools", "readthedocs", "devdocs", "visual studio code", "vs code",
            "pycharm", "intellij", "webstorm", "android studio", "xcode", "jupyter",
            "google colab", "colab.research", "overleaf", "notion.so", "obsidian",
            "onenote", "google docs", "docs.google", "sheets.google", "slides.google",
            "google scholar", "scholar.google", "arxiv", "researchgate",
            "sciencedirect", "jstor", "pubmed", "ieee", "desmos", "wolframalpha",
            "wolfram alpha", "geogebra", "symbolab", "khan academy", "khanacademy",
            "coursera", "edx.org", "udemy", "quizlet", "brilliant.org", "canvas",
            "blackboard", "moodle", "gradescope", "piazza", "instructure", "chegg",
            "wikipedia", "chatgpt", "claude.ai", "gemini.google", "copilot",
            "perplexity", "jira", "trello", "asana", "linear.app", "confluence",
            "figma", "miro.com", "lucidchart", "autocad", "solidworks",
        ]);

        this.dev_activity = new Set([
            "def ", "import ", "console.log", "printf(", "println", "#include",
            "std::", "public static void", "void main", "system.out", "fn main",
            "git commit", "git push", "git status", "git checkout", "git pull",
            "git merge", "git rebase", "git clone", "npm run", "npm install",
            "npm start", "pip install", "python -m", "cargo build", "cargo run",
            "docker build", "docker run", "kubectl ", "select * from",
            "insert into", "create table", "stack trace", "syntaxerror",
            "nameerror", "typeerror:", "segmentation fault", "npm err",
        ]);

        this.youtube_tokens = ["youtube", "youtu.be", " - youtube"];
        this.yt_educational = new Set([
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
        ]);
        this.yt_entertainment = new Set([
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
        ]);

        this._productiveMatcher = this._compileMatcher(this.productive_terms);
        this._distractedMatcher = this._compileMatcher(this.distracted_terms);

        this.customProductive = [];
        this.customDistracted = [];
    }

    setLabels(productive, distracted) {
        this.customProductive = productive || [];
        this.customDistracted = distracted || [];
    }

    reset() { this.state_history = []; }

    _compileMatcher(terms) {
        const wordTerms = [], substrTerms = [];
        for (const t of terms) {
            if (/^[a-z]{2,}$/.test(t)) wordTerms.push(escapeRegExp(t));
            else substrTerms.push(t);
        }
        const pattern = wordTerms.length ? new RegExp("\\b(?:" + wordTerms.join("|") + ")\\b", "g") : null;
        return { pattern, substrTerms };
    }

    _countHits(lowerText, matcher) {
        let hits = 0;
        if (matcher.pattern) {
            const found = lowerText.match(matcher.pattern);
            if (found) hits += new Set(found).size;
        }
        for (const term of matcher.substrTerms) if (lowerText.includes(term)) hits += 1;
        return hits;
    }

    _containsAny(lowerText, labels) {
        for (const label of labels) if (label && lowerText.includes(label.toLowerCase())) return true;
        return false;
    }

    _siteScore(lowerText, headerLower, sites) {
        let score = 0;
        for (const s of sites) {
            if (headerLower.includes(s)) score += 2;
            else if (lowerText.includes(s)) score += 1;
        }
        return score;
    }

    _cleanText(fullText) {
        const out = [];
        for (const w of fullText.split(/\s+/)) {
            const wl = w.toLowerCase();
            if (this.ignore_terms.has(wl)) continue;
            if (w.length <= 2 && !/^[a-z0-9]+$/i.test(w)) continue;
            out.push(w);
        }
        return out.join(" ");
    }

    _looksLikeDashboard(lowerText) {
        let hits = 0;
        for (const term of this.ignore_terms) if (lowerText.includes(term)) hits += 1;
        return hits >= 3;
    }

    _extractYoutubeTitle(headerLower) {
        let t = headerLower;
        for (const junk of [" - youtube", "youtube", "- google chrome", "google chrome",
            "mozilla firefox", "microsoft edge", "- safari", "safari", "- brave",
            "brave", "- opera", "arc"]) t = t.split(junk).join(" ");
        t = t.replace(/\(\d+\)/g, " ").replace(/\s+/g, " ").trim();
        return t;
    }

    _classifyYoutube(headerLower, lowerText) {
        const title = this._extractYoutubeTitle(headerLower);
        const probe = (title + " " + lowerText.slice(0, 500)).trim();
        let edu = 0, ent = 0;
        for (const t of this.yt_educational) if (probe.includes(t)) edu += 1;
        for (const t of this.yt_entertainment) if (probe.includes(t)) ent += 1;
        edu += this._countHits(title, this._productiveMatcher);
        ent += this._countHits(title, this._distractedMatcher);

        if (edu && !ent) return "Productive";
        if (ent && !edu) return "Distracted";
        if (edu >= ent + 2) return "Productive";
        if (ent >= edu + 2) return "Distracted";

        // Ambiguous: no zero-shot model in the browser. Lean on any weak signal,
        // and default a bare/unreadable YouTube view to Distracted (matches the
        // desktop fallback for titles with no educational signal).
        const text = (title || lowerText.slice(0, 300)).trim();
        if (text.length < 4) return "Distracted";
        if (edu > ent) return "Productive";
        if (ent > edu) return "Distracted";
        return "Distracted";
    }

    _getRawState(fullText, headerText) {
        const cleaned = this._cleanText(fullText);
        const lowerText = cleaned.toLowerCase();
        const headerLower = (headerText || "").toLowerCase();

        if (lowerText.trim().length < 10) return "Neutral";

        // 1. Custom labels win (distracted checked first).
        if (this._containsAny(lowerText, this.customDistracted)) return "Distracted";
        if (this._containsAny(lowerText, this.customProductive)) return "Productive";

        // 2. Site / app identity dominates.
        const dSite = this._siteScore(lowerText, headerLower, this.distracting_sites);
        const pSite = this._siteScore(lowerText, headerLower, this.productive_sites);
        const youtubeOpen = this._containsAny(headerLower, this.youtube_tokens) ||
            (lowerText.includes("youtube") &&
                this._containsAny(lowerText, ["subscribe", "subscribers", "up next", "views", "watch later", "comments"]));

        if (pSite > dSite && pSite > 0) return "Productive";
        if (dSite > 0 && dSite >= pSite) return "Distracted";
        if (youtubeOpen) return this._classifyYoutube(headerLower, lowerText);

        // 3. Unambiguous coding / CLI activity.
        if (this._containsAny(lowerText, this.dev_activity)) return "Productive";
        for (const marker of this.coding_markers) if (marker.test(fullText)) return "Productive";
        const totalChars = lowerText.replace(/ /g, "").length;
        if (totalChars >= 20) {
            let special = 0;
            for (const c of lowerText) if ("{}[]=;<>".includes(c)) special += 1;
            if (special / totalChars > 0.08) return "Productive";
        }

        // 4. Weighted general keywords (header double-counted), ties -> Distracted.
        const productiveHits = this._countHits(lowerText, this._productiveMatcher) + this._countHits(headerLower, this._productiveMatcher);
        const distractedHits = this._countHits(lowerText, this._distractedMatcher) + this._countHits(headerLower, this._distractedMatcher);
        if (productiveHits || distractedHits) return productiveHits > distractedHits ? "Productive" : "Distracted";

        // 5. No zero-shot fallback in-browser: a text-heavy screen with no
        //    decisive signal is treated as active work rather than collapsing to
        //    Neutral; only a sparse screen stays Neutral.
        return lowerText.length >= 200 ? "Productive" : "Neutral";
    }

    classify(fullText, headerText) {
        const rawLower = fullText.toLowerCase();
        let rawState;
        if (this._looksLikeDashboard(rawLower)) rawState = "Neutral";
        else rawState = this._getRawState(fullText, headerText);

        this.state_history.push(rawState);
        if (this.state_history.length > this.history_len) this.state_history.shift();

        const counts = {};
        let best = this.state_history[0], bestN = 0;
        for (const s of this.state_history) { counts[s] = (counts[s] || 0) + 1; if (counts[s] > bestN) { bestN = counts[s]; best = s; } }
        return best;
    }
}

/* ========================================================================== */
/*  ScreenEngine — getDisplayMedia + Tesseract.js                              */
/* ========================================================================== */
class ScreenEngine {
    constructor() {
        this.SCREEN_INTERVAL_SEC = 8;
        this.classifier = new ScreenClassifier();
        this.video = document.createElement("video");
        this.video.muted = true;
        this.canvas = document.createElement("canvas");
        this.stream = null;
        this.worker = null;
        this.running = false;
        this.state = "Neutral";
        this._busy = false;
    }

    isSharing() { return !!this.stream; }

    async start() {
        this.stream = await navigator.mediaDevices.getDisplayMedia({
            video: { frameRate: 1 },
            audio: false,
        });
        // If the user stops sharing via the browser's own control, clean up.
        this.stream.getVideoTracks()[0].addEventListener("ended", () => this.stop());
        this.video.srcObject = this.stream;
        await this.video.play();

        if (!this.worker) {
            // eslint-disable-next-line no-undef
            this.worker = await Tesseract.createWorker("eng");
        }
        this.classifier.reset();
        this.running = true;
        this._loop();
    }

    stop() {
        this.running = false;
        if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
        this.video.srcObject = null;
        this.state = "Screen Off";
    }

    setLabels(p, d) { this.classifier.setLabels(p, d); }

    async _grabAndClassify() {
        const vw = this.video.videoWidth, vh = this.video.videoHeight;
        if (!vw || !vh) return;
        const maxWidth = 1600;
        const scale = vw > maxWidth ? maxWidth / vw : 1;
        const cw = Math.round(vw * scale), ch = Math.round(vh * scale);
        this.canvas.width = cw;
        this.canvas.height = ch;
        const ctx = this.canvas.getContext("2d");
        ctx.drawImage(this.video, 0, 0, cw, ch);

        // Request blocks so per-word boxes/confidence are populated (Tesseract v5
        // omits them by default). Fall back to flattening blocks if the flat
        // `words` array isn't present.
        const { data } = await this.worker.recognize(this.canvas, {}, { blocks: true });
        let words = data.words;
        if (!words || !words.length) words = flattenWords(data);

        const headerCut = ch * 0.12;
        const headerTokens = [], bodyTokens = [];
        for (const word of words) {
            if ((word.confidence || 0) <= 30) continue;
            const yTop = word.bbox ? word.bbox.y0 : 0;
            (yTop < headerCut ? headerTokens : bodyTokens).push(word.text);
        }
        const fullText = headerTokens.concat(bodyTokens).join(" ");
        const headerText = headerTokens.join(" ");
        this.state = this.classifier.classify(fullText, headerText);
    }

    async _loop() {
        while (this.running) {
            if (!this._busy) {
                this._busy = true;
                try { await this._grabAndClassify(); }
                catch (e) { /* OCR hiccup; keep last state */ }
                this._busy = false;
            }
            await sleep(this.SCREEN_INTERVAL_SEC * 1000);
        }
    }
}

/* ========================================================================== */
/*  AppState — port of app/state_manager.py                                    */
/* ========================================================================== */
const HISTORY_KEY = "mindlens-history-v2";

class AppState {
    constructor() {
        this.is_running = false;
        this.session_started_at = null;
        this.camera_enabled = true;
        this.screen_enabled = true;
        this.camera_state = "Idle";
        this.screen_state = "Neutral";
        this.unified_state = "Neutral";
        this.timers = { productive: 0, distracted: 0, neutral: 0, total: 0 };
        this.custom_productive = [];
        this.custom_distracted = [];
        this.override_state = null;
        this.override_until = 0;
        this.session_history = this._loadHistory();
    }

    _loadHistory() {
        try {
            const data = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
            return Array.isArray(data) ? data : [];
        } catch (_) { return []; }
    }

    _saveHistory() {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(this.session_history.slice(-50)));
    }

    _overrideActive() {
        return this.override_state !== null && Date.now() / 1000 < this.override_until;
    }

    effectiveState(detected) {
        return this._overrideActive() ? this.override_state : detected;
    }

    setOverride(state, durationSec = 300) {
        if (!["Productive", "Distracted", "Neutral"].includes(state)) return false;
        this.override_state = state;
        this.override_until = Date.now() / 1000 + durationSec;
        return true;
    }

    clearOverride() { this.override_state = null; this.override_until = 0; }

    resetTimers() {
        this.timers = { productive: 0, distracted: 0, neutral: 0, total: 0 };
        this.override_state = null;
        this.override_until = 0;
        this.session_started_at = new Date().toISOString();
    }

    stopSession() {
        this.is_running = false;
        this.override_state = null;
        this.override_until = 0;
        const totalSeconds = Math.floor(this.timers.total);
        if (totalSeconds <= 0) return null;
        const entry = {
            id: new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14),
            started_at: this.session_started_at,
            ended_at: new Date().toISOString(),
            unified_state: this.unified_state,
            timers: {
                productive: secToHMS(this.timers.productive),
                distracted: secToHMS(this.timers.distracted),
                neutral: secToHMS(this.timers.neutral),
                total: secToHMS(this.timers.total),
            },
            seconds: {
                productive: Math.floor(this.timers.productive),
                distracted: Math.floor(this.timers.distracted),
                neutral: Math.floor(this.timers.neutral),
                total: totalSeconds,
            },
        };
        this.session_history.push(entry);
        this.session_history = this.session_history.slice(-50);
        this._saveHistory();
        return entry;
    }

    getHistory() { return this.session_history.slice().reverse(); }

    getStatusDict() {
        const overrideActive = this._overrideActive();
        return {
            is_running: this.is_running,
            session_started_at: this.session_started_at,
            camera_enabled: this.camera_enabled,
            screen_enabled: this.screen_enabled,
            camera_state: this.camera_state,
            screen_state: this.screen_state,
            unified_state: this.unified_state,
            override_state: overrideActive ? this.override_state : null,
            override_remaining: overrideActive ? Math.max(0, Math.floor(this.override_until - Date.now() / 1000)) : 0,
            timers: {
                productive: secToHMS(this.timers.productive),
                distracted: secToHMS(this.timers.distracted),
                neutral: secToHMS(this.timers.neutral),
                total: secToHMS(this.timers.total),
            },
            history_count: this.session_history.length,
            custom_productive: this.custom_productive,
            custom_distracted: this.custom_distracted,
        };
    }
}

/* ========================================================================== */
/*  Engine controller — ties it together, drives the status callback           */
/* ========================================================================== */
export function createEngine({ video, overlay, onStatus, onError }) {
    const state = new AppState();
    const camera = new CameraEngine(video, overlay);
    const screen = new ScreenEngine();

    let tickTimer = null;
    let emitTimer = null;
    let lastTick = 0;

    function emit() { onStatus(state.getStatusDict()); }

    function tickTimers() {
        const now = performance.now();
        const dt = lastTick ? (now - lastTick) / 1000 : 0;
        lastTick = now;
        if (!state.is_running) return;

        state.camera_state = state.camera_enabled ? camera.state : "Camera Off";
        state.screen_state = state.screen_enabled && screen.isSharing() ? screen.state : "Screen Off";
        const detected = getUnifiedState(state.camera_state, state.screen_state);
        state.unified_state = state.effectiveState(detected);

        if (state.unified_state === "Distracted") state.timers.distracted += dt;
        else if (state.unified_state === "Productive") state.timers.productive += dt;
        else state.timers.neutral += dt;
        state.timers.total += dt;
    }

    return {
        state,
        getStatus: () => state.getStatusDict(),

        async startSession() {
            state.resetTimers();
            state.is_running = true;
            state.camera_state = "Initializing";
            state.screen_state = state.screen_enabled ? "Initializing" : "Screen Off";
            emit();

            if (state.camera_enabled) {
                try { await camera.start(); }
                catch (e) { state.camera_state = "Camera Unavailable"; if (onError) onError("camera", e); }
            }
            lastTick = performance.now();
            tickTimer = setInterval(tickTimers, 250);
            emitTimer = setInterval(emit, 500);
            emit();
        },

        stopSession() {
            clearInterval(tickTimer); clearInterval(emitTimer);
            tickTimer = emitTimer = null;
            camera.stop();
            screen.stop();
            const entry = state.stopSession();
            state.camera_state = "Idle";
            state.screen_state = "Neutral";
            state.unified_state = "Neutral";
            emit();
            return entry;
        },

        async enableCamera(on) {
            state.camera_enabled = on;
            if (state.is_running) {
                if (on && !camera.running) {
                    state.camera_state = "Initializing"; emit();
                    try { await camera.start(); } catch (e) { state.camera_state = "Camera Unavailable"; }
                } else if (!on && camera.running) {
                    camera.stop();
                }
            }
            emit();
        },

        async enableScreen(on) {
            state.screen_enabled = on;
            if (state.is_running && on && !screen.isSharing()) {
                try { await screen.start(); }
                catch (e) { state.screen_enabled = false; if (onError) onError("screen", e); }
            } else if (!on && screen.isSharing()) {
                screen.stop();
            }
            emit();
        },

        // Screen capture needs an explicit user gesture (the browser picker), so
        // it's wired to its own button rather than auto-starting with the session.
        async shareScreen() {
            if (!state.screen_enabled) state.screen_enabled = true;
            try { await screen.start(); emit(); return true; }
            catch (e) { if (onError) onError("screen", e); return false; }
        },
        stopScreen() { screen.stop(); emit(); },
        isScreenSharing: () => screen.isSharing(),

        setOverride(s) { state.setOverride(s); emit(); },
        clearOverride() { state.clearOverride(); emit(); },

        setLabels(productive, distracted) {
            state.custom_productive = productive;
            state.custom_distracted = distracted;
            screen.setLabels(productive, distracted);
        },
    };
}

/* -------------------------------------------------------------------------- */
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
// Flatten Tesseract block -> paragraph -> line -> word into a flat word list.
function flattenWords(data) {
    const words = [];
    for (const block of data.blocks || []) {
        for (const para of block.paragraphs || []) {
            for (const line of para.lines || []) {
                for (const w of line.words || []) words.push(w);
            }
        }
    }
    return words;
}
function escapeRegExp(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
function secToHMS(totalSeconds) {
    const total = Math.floor(totalSeconds);
    return { h: Math.floor(total / 3600), m: Math.floor((total % 3600) / 60), s: total % 60 };
}
