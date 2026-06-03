import pyautogui
import cv2
import re
import numpy as np
import easyocr
import torch
import time
import threading
from transformers import pipeline

class ScreenAnalyzer:
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
            
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=self.device)
        
        self.MACRO_LABELS = ["software engineering or studying", "entertainment or social media", "neutral desktop background"]
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
            # --- Code Structure & Languages ---
            "def ", "import ", "const ", "let ", "var ", "console.log", "public class", 
            "fn ", "impl ", "std::", "iostream", "#include", "printf(", "scanf(", 
            "using namespace", "public static void main", "System.out.println", 
            "Console.WriteLine", "try {", "catch (", "finally {", "async function", 
            "await ", "yield ", "lambda ", "struct ", "interface ", "enum ", 
            "extends ", "implements ", "export default", "module.exports",
            
            # --- Database & SQL ---
            "select * from", "insert into", "update set", "delete from", "join ", 
            "inner join", "group by", "order by", "create table", "drop table", 
            "alter table", "document.get", "db.collection", "mongoose.connect",
            
            # --- CLI & Git Commands ---
            "git commit", "git push", "git checkout", "git clone", "git status",
            "git pull", "git merge", "git rebase", "git branch", "git log",
            "npm run", "npm install", "npm start", "npm test", "yarn add", "yarn start",
            "pip install", "pip freeze", "conda activate", "python -m", "cargo build", 
            "cargo run", "make build", "gcc ", "g++ ", "cmake ", "mvn clean", "gradlew", 
            "docker run", "docker build", "docker-compose up", "kubectl ", "terraform ",
            
            # --- Config & Files ---
            "package.json", "cargo.toml", "dockerfile", "docker-compose.yml", 
            "requirements.txt", "pom.xml", "build.gradle", "tsconfig.json", 
            ".gitignore", "webpack.config",
            
            # --- Dev Tools & Sites ---
            "localhost:", "127.0.0.1:", "8080", "3000", "5000", "stack overflow", 
            "stackoverflow.com", "github.com", "gitlab.com", "bitbucket.org", 
            "leetcode.com", "hackerrank.com", "geeksforgeeks.org", "mdn web docs", 
            "developer.mozilla.org", "w3schools.com", "freecodecamp.org", "codecademy.com", 
            "kaggle.com", "huggingface.co", "vs code", "visual studio code", "pycharm", 
            "intellij", "webstorm", "cursor editor", "jupyter notebook", "google colab", 
            "copilot", "chatgpt", "claude.ai", "gemini.google", "postman", "insomnia",
            
            # --- Math / Science / Design ---
            "calculator", "desmos.com", "wolframalpha.com", "wolfram alpha", 
            "geogebra.org", "symbolab.com", "mathway.com", "overleaf.com", "arxiv.org",
            "trello.com", "jira.com", "figma.com", "autocad", "solidworks", "lucidchart.com",
            
            # --- Studying & Academia ---
            "quizlet.com", "khan academy", "coursera.org", "blackboard.com", 
            "canvas.instructure", "edx.org", "udemy.com", "pluralsight.com", 
            "datacamp.com", "skillshare.com", "google scholar", "researchgate.net", 
            "moodle", "turnitin"
        }

        self.ignore_terms = {
            # --- Original App Terms ---
            "mindlens", "live focus workspace", "focus score", "saved sessions",
            "camera feed", "session analytics", "previous sessions", "custom labels",
            "camera detection", "screen detection", "productive", "distracted",
            "neutral", "total session", "override detection",
            
            # --- Extended App UI & States ---
            "dashboard", "settings", "preferences", "user profile", "account info",
            "sign in", "sign out", "login", "logout", "billing", "subscription",
            "upgrade to pro", "help center", "customer support", "terms of service",
            "privacy policy", "notifications", "app.mindlens", "mindlens.com",
            "focus mode active", "distraction blocked", "tracker daemon", "agent status"
        }

        self.productive_terms = {
            # --- Languages & Frameworks ---
            "python", "javascript", "typescript", "java", "c++", "c#", "rust", 
            "golang", "ruby", "php", "swift", "kotlin", "scala", "dart", "react", 
            "angular", "vue", "svelte", "next.js", "flask", "django", "spring boot", 
            "laravel", "ruby on rails", "react native", "flutter",
            
            # --- Libraries & Tech Concepts ---
            "numpy", "pandas", "opencv", "mediapipe", "transformers", "pytorch", 
            "tensorflow", "scikit-learn", "api", "function", "class", "import", "def ", 
            "const ", "return", "git", "docker", "kubernetes", "aws", "azure", "gcp", 
            "sql", "postgres", "mysql", "mongodb", "redis", "elasticsearch", "linux", 
            "ubuntu", "terminal", "powershell", "bash", "vim", "neovim", "documentation", "docs",
            
            # --- Dev Tools & Editors ---
            "github", "stackoverflow", "stack overflow", "visual studio code", "vs code",
            "cursor", "pycharm", "sublime", "intellij", "eclipse", "xcode", "webstorm",
            "postman", "wireshark", "developer", "programming", "coding", "compiler", "debugger",
            
            # --- Math, Science & Engineering ---
            "calculator", "calc", "math", "algebra", "geometry", "calculus", "desmos",
            "wolfram", "statistics", "equation", "theorem", "formula", "physics", 
            "chemistry", "biology", "economics", "finance", "accounting", "anatomy", 
            "thermodynamics", "mechanics", "matrix", "integral", "derivative", "graphing", 
            "machine learning", "artificial intelligence", "neural network", "deep learning",
            
            # --- Design & Creative ---
            "figma", "adobe xd", "sketch", "photoshop", "illustrator", "premiere pro",
            "after effects", "blender", "unity", "unreal engine", "wireframe", "ui/ux",
            
            # --- School & Academia ---
            "notion", "canvas", "blackboard", "quizlet", "khan academy", "coursera",
            "leetcode", "overleaf", "latex", "paper", "research", "assignment",
            "homework", "lecture", "syllabus", "textbook", "study guide", "essay",
            "thesis", "dissertation", "presentation", "report", "proposal",
            
            # --- General Office & Productivity ---
            "excel", "sheets", "powerpoint", "slides", "word", "pages", "numbers",
            "slack", "teams", "jira", "trello", "asana", "monday.com", "zoom", 
            "meet", "meeting", "calendar", "notes", "pdf", "preview", "drive", 
            "dropbox", "evernote", "obsidian", "onenote", "resume", "cv", "cover letter",
            "email", "gmail", "outlook", "inbox", "schedule", "planner", "todo"
        }

        self.distracted_terms = {
            # --- Video & Streaming ---
            "youtube", "netflix", "hulu", "disney+", "prime video", "hbo max", 
            "peacock", "paramount+", "apple tv", "crunchyroll", "funimation", 
            "vudu", "tubi", "vimeo", "dailymotion", "twitch",
            
            # --- Social Media ---
            "tiktok", "instagram", "reddit", "twitter", "x.com", "facebook", 
            "snapchat", "pinterest", "tumblr", "bereal", "threads", "bluesky", 
            "mastodon", "weibo", "discord", "whatsapp", "telegram", "wechat",
            
            # --- Gaming Platforms & Games ---
            "game", "gaming", "roblox", "minecraft", "fortnite", "steam", 
            "epic games", "battle.net", "origin", "ubisoft", "league of legends", 
            "valorant", "cs:go", "cs2", "dota 2", "apex legends", "call of duty", 
            "warzone", "overwatch", "genshin impact", "honkai", "xbox", 
            "playstation", "nintendo", "ign", "kotaku", "polygon",
            
            # --- Content & Trends ---
            "memes", "shorts", "reels", "buzzfeed", "imgur", "9gag", "4chan",
            
            # --- Music & Audio ---
            "spotify", "apple music", "soundcloud", "pandora", "tidal", "last.fm",
            
            # --- Shopping & Ecommerce ---
            "amazon.com", "ebay", "walmart", "target", "etsy", "aliexpress", 
            "temu", "shein", "craigslist", "zillow", "wayfair", "asos",
            
            # --- News, Gossip & Sports ---
            "tmz", "daily mail", "fox news", "cnn", "nytimes", "huffpost", 
            "espn", "bleacher report", "fantasy football", "sports betting", "draftkings",
            
            # --- Dating ---
            "tinder", "bumble", "hinge", "okcupid", "match.com"
        }
        self.current_state = "Neutral"
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run_loop(self):
        print("[Screen] Analyzer background thread started...")
        while self.running:
            try:
                if self.enabled_provider and not self.enabled_provider():
                    time.sleep(1)
                    continue

                screenshot = pyautogui.screenshot()
                screenshot_array = np.array(screenshot)
                img = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2GRAY)
                height, width = img.shape
                max_width = 1600
                if width > max_width:
                    scale = max_width / width
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                raw_results = self.reader.readtext(img, detail=1)
                
                text_list = [res[1] for res in raw_results if res[2] > 0.3]
                full_text = " ".join(text_list)
                self.current_state = self._classify_text(full_text)
                    
                time.sleep(10)
                
            except Exception as e:
                print(f"[Screen] Error encountered: {e}")
                time.sleep(10)

    def get_state(self):
        return self.current_state

    def _classify_text(self, full_text):
        raw_lower_text = full_text.lower()
        if self._looks_like_mindlens_dashboard(raw_lower_text):
            raw_state = "Neutral"
        else:
            raw_state = self._get_raw_state(full_text)

        # Apply state history smoothing (hysteresis)
        self.state_history.append(raw_state)
        if len(self.state_history) > self.history_len:
            self.state_history.pop(0)

        # Return majority vote
        return max(set(self.state_history), key=self.state_history.count)

    def _get_raw_state(self, full_text):
        cleaned_text = self._clean_text(full_text)
        lower_text = cleaned_text.lower()

        if len(lower_text.strip()) < 10:
            return "Neutral"

        # 1. Custom overrides from the UI
        productive_labels, distracted_labels = self._custom_labels()
        if self._contains_any(lower_text, productive_labels):
            return "Productive"
        if self._contains_any(lower_text, distracted_labels):
            return "Distracted"

        # 2. Strong productive keyword & regex code/dev markers short-circuit
        if self._contains_any(lower_text, self.strong_productive_keywords):
            return "Productive"

        for marker in self.coding_markers:
            if re.search(marker, full_text):
                return "Productive"

        # Check for density of typical code symbols
        total_chars = len(lower_text.replace(" ", ""))
        if total_chars >= 20:
            special_symbols = sum(1 for c in lower_text if c in "{}[]=;<>")
            if special_symbols / total_chars > 0.08:
                return "Productive"

        # 3. Math / Calculator / Spreadsheet Heuristic
        # Check digit/operator density
        if total_chars >= 5:
            digit_op_chars = sum(1 for c in lower_text if c.isdigit() or c in "+-*/=")
            if digit_op_chars / total_chars > 0.35:
                return "Productive"

        # 4. Hit count check for weaker/general keywords
        productive_hits = self._count_hits(lower_text, self.productive_terms)
        distracted_hits = self._count_hits(lower_text, self.distracted_terms)
        if productive_hits >= 1 and productive_hits >= distracted_hits:
            return "Productive"
        if distracted_hits >= 1 and distracted_hits > productive_hits:
            return "Distracted"

        # 5. Fallback to Zero-Shot Classification
        result = self.classifier(
            cleaned_text[:1500],
            self.MACRO_LABELS,
            hypothesis_template="The content of this screen is related to {}."
        )

        top_label = result['labels'][0]
        top_score = result['scores'][0]
        second_score = result['scores'][1] if len(result['scores']) > 1 else 0.0

        if top_label == "software engineering or studying":
            if top_score >= 0.45 and top_score - second_score >= 0.08:
                return "Productive"
        elif top_label == "entertainment or social media":
            if top_score >= 0.60 and top_score - second_score >= 0.08:
                return "Distracted"

        return "Neutral"

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

    def _count_hits(self, lower_text, terms):
        return sum(1 for term in terms if term in lower_text)
