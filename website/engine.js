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
// transformers.js (v3, WebGPU-capable) powers the on-device CLIP screen model.
const TRANSFORMERS_URL =
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

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
        this.state = "Idle";        // object/pose state (phone / study / read) — drives scoring
        this.action = "Idle";       // richer presence action for display + metrics
        this.facePresent = false;
        this.cachedObjects = [];
        this.cachedObjectsTs = 0;
        this.TARGET_AI_FPS = 8;
        this.BOX_STALE_SEC = 0.8;
        this._loadingPromise = null;
        this._lastVideoTs = -1;
        this._mouthHistory = [];     // {t, open} for speaking detection
        this._faceMissingSince = 0;
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
        this.action = "Camera Off";
        this.facePresent = false;
        this._mouthHistory = [];
        this._faceMissingSince = 0;
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
        let facePresent = false;
        let yawRatio = 0;   // -1..1, magnitude = how far the head is turned
        let mouthOpen = 0;  // normalized lip separation

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
            facePresent = true;
            const face = faceRes.faceLandmarks[0];
            const forehead = face[10], chin = face[152], nose = face[1];
            chinY = chin.y * vh;
            const faceHeight = Math.abs(forehead.y - chin.y);
            if (faceHeight > 0) {
                const ratio = (chin.y - nose.y) / faceHeight;
                if (ratio < 0.36) headDown = true;
            }
            // Head yaw: nose x offset from the midpoint of the two cheek edges,
            // normalized by half the face width. Large magnitude = looking aside.
            const rCheek = face[234], lCheek = face[454];
            const center = (rCheek.x + lCheek.x) / 2;
            const halfW = Math.abs(lCheek.x - rCheek.x) / 2 || 1;
            yawRatio = (nose.x - center) / halfW;
            // Mouth openness (upper vs lower inner lip), normalized by face height.
            const upperLip = face[13], lowerLip = face[14];
            if (faceHeight > 0) mouthOpen = Math.abs(lowerLip.y - upperLip.y) / faceHeight;
        }
        return { handCoords, isPinching, headDown, chinY, facePresent, yawRatio, mouthOpen };
    }

    // Merge the object/pose state with presence cues into a display action.
    _updateAction(objectState, pose) {
        const now = performance.now() / 1000;

        // Speaking: sustained mouth-openness variation over a short window.
        this._mouthHistory.push({ t: now, open: pose.mouthOpen });
        while (this._mouthHistory.length && this._mouthHistory[0].t < now - 1.5) this._mouthHistory.shift();
        let speaking = false;
        if (pose.facePresent && this._mouthHistory.length >= 4) {
            const opens = this._mouthHistory.map((h) => h.open);
            if (Math.max(...opens) - Math.min(...opens) > 0.035) speaking = true;
        }

        // Absence: require a couple of seconds of no face to call it "Away".
        if (!pose.facePresent) {
            if (!this._faceMissingSince) this._faceMissingSince = now;
        } else {
            this._faceMissingSince = 0;
        }
        const away = this._faceMissingSince && now - this._faceMissingSince > 2;
        const lookingAway = pose.facePresent && Math.abs(pose.yawRatio) > 0.55;

        this.facePresent = pose.facePresent;

        // Precedence, most-actionable first.
        if (objectState === "Actively Using Phone") this.action = "On phone";
        else if (away) this.action = "Away";
        else if (lookingAway) this.action = "Looking away";
        else if (objectState === "Studying / Writing") this.action = "Writing";
        else if (objectState === "Reading") this.action = "Reading";
        else if (speaking) this.action = "Speaking";
        else if (pose.facePresent) this.action = "Focused";
        else this.action = "Idle";
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
                    const pose = this._pose(ts);
                    this.state = this.analyzer.analyze(objects, pose.handCoords, pose.isPinching, pose.headDown, pose.chinY);
                    this._updateAction(this.state, pose);
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

        // ── Activity categories ─────────────────────────────────────────────
        // Used to report a granular activity (Coding, Meeting, Email…) on top of
        // the productive/distracted verdict. Checked against OCR'd site/app text.
        this.meeting_sites = new Set([
            "zoom.us", "zoom meeting", "meet.google", "google meet", "teams.microsoft",
            "microsoft teams", "webex", "gotomeeting", "whereby", "bluejeans",
            "join meeting", "leave meeting", "mute", "unmute", "start video", "stop video",
            "participants", "raise hand", "share screen", "you are muted", "recording…",
        ]);
        this.coding_sites = new Set([
            "github", "gitlab", "bitbucket", "stack overflow", "stackoverflow", "leetcode",
            "hackerrank", "codeforces", "geeksforgeeks", "codecademy", "freecodecamp",
            "replit", "codepen", "codesandbox", "jsfiddle", "visual studio code", "vs code",
            "pycharm", "intellij", "webstorm", "android studio", "xcode", "jupyter",
            "google colab", "developer.mozilla", "mdn web docs", "w3schools", "readthedocs",
        ]);
        this.writing_sites = new Set([
            "google docs", "docs.google", "notion.so", "notion", "obsidian", "onenote",
            "overleaf", "word", "pages", "confluence", "quip", "dropbox paper",
        ]);
        this.email_sites = new Set([
            "gmail", "outlook", "mail.google", "proton mail", "protonmail", "yahoo mail",
            "inbox", "compose", "outlook.office",
        ]);
        this.video_sites = new Set([
            "youtube", "netflix", "hulu", "disney+", "disneyplus", "prime video", "hbo max",
            "max.com", "peacock", "paramount+", "twitch", "crunchyroll", "vimeo", "dailymotion",
        ]);
        this.social_sites = new Set([
            "tiktok", "instagram", "reddit", "twitter", "x.com", "facebook", "snapchat",
            "pinterest", "tumblr", "threads", "bluesky", "mastodon", "discord",
        ]);
        this.gaming_sites = new Set([
            "roblox", "minecraft", "fortnite", "steampowered", "steamcommunity", "epic games",
            "battle.net", "valorant", "league of legends", "genshin", "call of duty",
            "warzone", "overwatch", "apex legends", "xbox", "playstation",
        ]);
        this.shopping_sites = new Set([
            "amazon.com", "ebay", "aliexpress", "temu", "shein", "etsy", "walmart",
            "target.com", "wayfair", "asos", "add to cart", "add to bag", "checkout",
        ]);

        this._productiveMatcher = this._compileMatcher(this.productive_terms);
        this._distractedMatcher = this._compileMatcher(this.distracted_terms);

        this.customProductive = [];
        this.customDistracted = [];
        this._lastReason = "none";
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

        if (lowerText.trim().length < 10) { this._lastReason = "sparse"; return "Neutral"; }

        // 1. Custom labels win (distracted checked first).
        if (this._containsAny(lowerText, this.customDistracted)) { this._lastReason = "custom"; return "Distracted"; }
        if (this._containsAny(lowerText, this.customProductive)) { this._lastReason = "custom"; return "Productive"; }

        // 2. Site / app identity dominates.
        const dSite = this._siteScore(lowerText, headerLower, this.distracting_sites);
        const pSite = this._siteScore(lowerText, headerLower, this.productive_sites);
        const youtubeOpen = this._containsAny(headerLower, this.youtube_tokens) ||
            (lowerText.includes("youtube") &&
                this._containsAny(lowerText, ["subscribe", "subscribers", "up next", "views", "watch later", "comments"]));

        if (pSite > dSite && pSite > 0) { this._lastReason = "site"; return "Productive"; }
        if (dSite > 0 && dSite >= pSite) { this._lastReason = "site"; return "Distracted"; }
        if (youtubeOpen) { this._lastReason = "youtube"; return this._classifyYoutube(headerLower, lowerText); }

        // 3. Unambiguous coding / CLI activity.
        if (this._containsAny(lowerText, this.dev_activity)) { this._lastReason = "dev"; return "Productive"; }
        for (const marker of this.coding_markers) if (marker.test(fullText)) { this._lastReason = "dev"; return "Productive"; }
        const totalChars = lowerText.replace(/ /g, "").length;
        if (totalChars >= 20) {
            let special = 0;
            for (const c of lowerText) if ("{}[]=;<>".includes(c)) special += 1;
            if (special / totalChars > 0.08) { this._lastReason = "dev"; return "Productive"; }
        }

        // 4. Weighted general keywords (header double-counted), ties -> Distracted.
        const productiveHits = this._countHits(lowerText, this._productiveMatcher) + this._countHits(headerLower, this._productiveMatcher);
        const distractedHits = this._countHits(lowerText, this._distractedMatcher) + this._countHits(headerLower, this._distractedMatcher);
        if (productiveHits || distractedHits) { this._lastReason = "keywords"; return productiveHits > distractedHits ? "Productive" : "Distracted"; }

        // 5. No decisive text signal — leave the verdict to the vision model.
        this._lastReason = lowerText.length >= 200 ? "textheavy" : "sparse";
        return lowerText.length >= 200 ? "Productive" : "Neutral";
    }

    // Granular activity from OCR'd text + which branch decided the verdict.
    _activityFromText(lowerText, headerLower, state, reason) {
        const t = headerLower + " " + lowerText;
        const hit = (set) => { for (const s of set) if (t.includes(s)) return true; return false; };
        if (hit(this.meeting_sites)) return "Meeting";
        if (reason === "dev" || hit(this.coding_sites)) return "Coding";
        if (hit(this.email_sites)) return "Email";
        if (hit(this.writing_sites)) return "Writing";
        if (reason === "youtube") return state === "Productive" ? "Educational video" : "Watching video";
        if (hit(this.video_sites)) return "Watching video";
        if (hit(this.social_sites)) return "Social media";
        if (hit(this.gaming_sites)) return "Gaming";
        if (hit(this.shopping_sites)) return "Shopping";
        if (state === "Productive") return "Working";
        if (state === "Distracted") return "Distracted";
        return "Idle";
    }

    // One-shot OCR analysis (no smoothing — the ScreenEngine fuses + smooths).
    // Returns { state, activity, decisive }. `decisive` means the verdict came
    // from a high-precision signal (custom label / site identity / dev activity
    // / YouTube title) that should outrank the vision model.
    analyze(fullText, headerText) {
        const rawLower = fullText.toLowerCase();
        this._lastReason = "none";
        let state;
        if (this._looksLikeDashboard(rawLower)) { state = "Neutral"; this._lastReason = "dashboard"; }
        else state = this._getRawState(fullText, headerText);
        const cleanedLower = this._cleanText(fullText).toLowerCase();
        const activity = this._activityFromText(cleanedLower, (headerText || "").toLowerCase(), state, this._lastReason);
        const decisive = ["custom", "site", "dev", "youtube"].includes(this._lastReason);
        return { state, activity, decisive };
    }
}

/* ========================================================================== */
/*  ScreenVision — on-device CLIP zero-shot image classification               */
/*                                                                             */
/*  Recognizes *what a screen is* semantically (code editor, video call, docs, */
/*  YouTube, social, game…) rather than relying on OCR text alone. This is the  */
/*  piece that lets screen detection work on things OCR can't read — e.g. a     */
/*  video-call grid during an interview.                                        */
/* ========================================================================== */

// Prompt -> granular activity + productive/distracted mapping. CLIP scores the
// screenshot against each prompt; the highest wins.
const CLIP_LABELS = [
    { label: "a code editor or IDE showing source code", activity: "Coding", state: "Productive" },
    { label: "a coding interview or online technical assessment", activity: "Coding", state: "Productive" },
    { label: "a terminal or command-line console", activity: "Terminal", state: "Productive" },
    { label: "a video call or online meeting with webcam tiles", activity: "Meeting", state: "Productive" },
    { label: "software documentation or a technical article", activity: "Reading", state: "Productive" },
    { label: "a text document or word processor being written", activity: "Writing", state: "Productive" },
    { label: "an email inbox", activity: "Email", state: "Productive" },
    { label: "a spreadsheet with rows and columns", activity: "Spreadsheet", state: "Productive" },
    { label: "a slide presentation being edited", activity: "Slides", state: "Productive" },
    { label: "a design or diagramming tool", activity: "Design", state: "Productive" },
    { label: "a streaming video or YouTube player", activity: "Watching video", state: "Distracted" },
    { label: "a social media feed", activity: "Social media", state: "Distracted" },
    { label: "a video game being played", activity: "Gaming", state: "Distracted" },
    { label: "an online shopping store", activity: "Shopping", state: "Distracted" },
    { label: "an instant messaging chat app", activity: "Chatting", state: "Neutral" },
    { label: "an empty desktop or home screen", activity: "Idle", state: "Neutral" },
];

class ScreenVision {
    constructor() {
        this.classifier = null;
        this.ready = false;
        this.loading = false;
        this.failed = false;
        this.device = null;
        this._smallCanvas = document.createElement("canvas");
    }

    // status: 'off' | 'loading' | 'ready' | 'failed'
    get status() {
        if (this.ready) return "ready";
        if (this.loading) return "loading";
        if (this.failed) return "failed";
        return "off";
    }

    async load() {
        if (this.ready || this.loading) return;
        this.loading = true;
        try {
            const { pipeline, env } = await import(TRANSFORMERS_URL);
            env.allowLocalModels = false;
            const build = (device, dtype) => pipeline(
                "zero-shot-image-classification", "Xenova/clip-vit-base-patch32", { device, dtype });
            try {
                if (navigator.gpu) { this.classifier = await build("webgpu", "fp16"); this.device = "webgpu"; }
                else throw new Error("no-webgpu");
            } catch (_) {
                this.classifier = await build("wasm", "q8"); this.device = "wasm";
            }
            this.ready = true;
        } catch (e) {
            this.failed = true;
            console.warn("[MindLens] CLIP vision model unavailable; falling back to OCR only:", e);
        } finally {
            this.loading = false;
        }
    }

    // Classify a source <video>/<canvas>; returns { activity, state, score } or null.
    async classify(source, sw, sh) {
        if (!this.ready) return null;
        // Downscale to keep the data-url small; CLIP resizes to 224 internally.
        const w = 384, h = Math.max(1, Math.round((sh / sw) * 384));
        this._smallCanvas.width = w;
        this._smallCanvas.height = h;
        this._smallCanvas.getContext("2d").drawImage(source, 0, 0, w, h);
        const url = this._smallCanvas.toDataURL("image/jpeg", 0.85);
        const out = await this.classifier(url, CLIP_LABELS.map((l) => l.label),
            { hypothesis_template: "a screenshot of {}" });
        const top = Array.isArray(out) ? out[0] : out;
        if (!top) return null;
        const meta = CLIP_LABELS.find((l) => l.label === top.label) || {};
        return { activity: meta.activity || "Unknown", state: meta.state || "Neutral", score: top.score || 0 };
    }
}

/* ========================================================================== */
/*  ScreenEngine — getDisplayMedia + OCR + CLIP fusion                          */
/* ========================================================================== */
class ScreenEngine {
    constructor() {
        this.SCREEN_INTERVAL_SEC = 4;
        this.classifier = new ScreenClassifier();
        this.vision = new ScreenVision();
        this.video = document.createElement("video");
        this.video.muted = true;
        this.canvas = document.createElement("canvas");
        this.stripCanvas = document.createElement("canvas");
        this.stream = null;
        this.worker = null;
        this.running = false;
        this.state = "Neutral";
        this.activity = "Idle";
        this.confidence = 0;
        this.source = "ocr";
        this._recent = [];
        this._busy = false;
    }

    isSharing() { return !!this.stream; }
    get visionStatus() { return this.isSharing() ? this.vision.status : "off"; }

    async start() {
        this.stream = await navigator.mediaDevices.getDisplayMedia({
            video: { frameRate: 2 },
            audio: false,
        });
        this.stream.getVideoTracks()[0].addEventListener("ended", () => this.stop());
        this.video.srcObject = this.stream;
        await this.video.play();

        if (!this.worker) {
            // eslint-disable-next-line no-undef
            this.worker = await Tesseract.createWorker("eng");
        }
        // Load the vision model in the background so OCR works immediately and
        // CLIP joins the fusion once it's downloaded/compiled.
        this.vision.load();
        this.classifier.reset();
        this._recent = [];
        this.running = true;
        this._loop();
    }

    stop() {
        this.running = false;
        if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
        this.video.srcObject = null;
        this.state = "Screen Off";
        this.activity = "Idle";
    }

    setLabels(p, d) { this.classifier.setLabels(p, d); }

    // OCR a canvas region into header/body token lists split at `headerFrac`.
    async _ocr(canvas, headerFrac) {
        const { data } = await this.worker.recognize(canvas, {}, { blocks: true });
        let words = data.words;
        if (!words || !words.length) words = flattenWords(data);
        const cut = canvas.height * headerFrac;
        const header = [], body = [];
        for (const word of words) {
            if ((word.confidence || 0) <= 30) continue;
            const yTop = word.bbox ? word.bbox.y0 : 0;
            (yTop < cut ? header : body).push(word.text);
        }
        return { header, body };
    }

    async _grabAndClassify() {
        const vw = this.video.videoWidth, vh = this.video.videoHeight;
        if (!vw || !vh) return;

        // Grayscale + boosted contrast makes Tesseract markedly more accurate on
        // UI text; applied to the OCR canvases only (CLIP still sees full colour).
        const OCR_FILTER = "grayscale(1) contrast(1.5) brightness(1.05)";

        // Full frame (downscaled) for body + a coarse header split.
        const maxWidth = 1600;
        const scale = vw > maxWidth ? maxWidth / vw : 1;
        const cw = Math.round(vw * scale), ch = Math.round(vh * scale);
        this.canvas.width = cw; this.canvas.height = ch;
        const fctx = this.canvas.getContext("2d");
        fctx.filter = OCR_FILTER;
        fctx.drawImage(this.video, 0, 0, cw, ch);

        // High-res, upscaled crop of the very top strip (browser tab bar / window
        // title) where the active site names itself — the most decisive signal.
        const stripH = Math.max(1, Math.round(vh * 0.08));
        const sw = Math.min(vw, 2200);
        this.stripCanvas.width = sw; this.stripCanvas.height = stripH * 2;
        const sctx = this.stripCanvas.getContext("2d");
        sctx.filter = OCR_FILTER;
        sctx.drawImage(this.video, 0, 0, vw, stripH, 0, 0, sw, stripH * 2);

        const [full, strip] = await Promise.all([
            this._ocr(this.canvas, 0.12),
            this._ocr(this.stripCanvas, 1.0), // whole strip is header
        ]);

        const headerTokens = strip.header.concat(full.header);
        const fullText = headerTokens.concat(full.body).join(" ");
        const headerText = headerTokens.join(" ");

        // OCR verdict (precise when decisive) + CLIP verdict (semantic).
        const ocr = this.classifier.analyze(fullText, headerText);
        let fused = { state: ocr.state, activity: ocr.activity, confidence: ocr.decisive ? 0.9 : 0.4, source: "ocr" };

        if (!ocr.decisive && this.vision.ready) {
            try {
                const v = await this.vision.classify(this.video, vw, vh);
                if (v) {
                    const agrees = v.state === ocr.state;
                    if (v.score >= 0.28) {
                        // Confident vision verdict wins; corroboration bumps confidence.
                        fused = { state: v.state, activity: v.activity,
                            confidence: agrees ? Math.min(0.99, v.score + 0.1) : v.score, source: "vision" };
                    } else if (ocr.state === "Neutral") {
                        // OCR had nothing; take the vision guess even if weak.
                        fused = { state: v.state, activity: v.activity, confidence: v.score, source: "vision-weak" };
                    } else if (v.score >= 0.18 && agrees) {
                        // Borderline vision that agrees with OCR's lean: keep the
                        // state, adopt the richer activity label.
                        fused = { state: ocr.state, activity: v.activity, confidence: Math.max(0.5, v.score), source: "fused" };
                    }
                }
            } catch (e) { /* vision hiccup; keep OCR verdict */ }
        }

        // Smooth over the last few readings (majority state; activity from the
        // most recent sample matching the winning state).
        this._recent.push(fused);
        if (this._recent.length > 4) this._recent.shift();
        const counts = {};
        let best = fused.state, bestN = 0;
        for (const r of this._recent) { counts[r.state] = (counts[r.state] || 0) + 1; if (counts[r.state] > bestN) { bestN = counts[r.state]; best = r.state; } }
        const latestMatch = [...this._recent].reverse().find((r) => r.state === best) || fused;

        this.state = best;
        this.activity = latestMatch.activity;
        this.confidence = latestMatch.confidence;
        this.source = latestMatch.source;
    }

    async _loop() {
        while (this.running) {
            if (!this._busy) {
                this._busy = true;
                try { await this._grabAndClassify(); }
                catch (e) { /* OCR/vision hiccup; keep last state */ }
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
        this.camera_action = "Idle";
        this.screen_state = "Neutral";
        this.screen_activity = "Idle";
        this.screen_confidence = 0;
        this.vision_status = "off";
        this.unified_state = "Neutral";
        this.timers = { productive: 0, distracted: 0, neutral: 0, total: 0 };
        this.custom_productive = [];
        this.custom_distracted = [];
        this.override_state = null;
        this.override_until = 0;
        this.metrics = this._freshMetrics();
        this.session_history = this._loadHistory();
    }

    _freshMetrics() {
        return {
            switches: 0,           // productive/distracted/neutral transitions
            away_seconds: 0,       // camera reported "Away"
            looking_away_seconds: 0,
            current_streak: 0,     // current continuous productive seconds
            longest_streak: 0,     // best productive streak this session
            _lastState: null,
        };
    }

    // Called every tick with elapsed dt and the live signals.
    updateMetrics(dt, unifiedState, cameraAction) {
        const m = this.metrics;
        if (m._lastState !== null && m._lastState !== unifiedState) m.switches += 1;
        m._lastState = unifiedState;

        if (cameraAction === "Away") m.away_seconds += dt;
        else if (cameraAction === "Looking away") m.looking_away_seconds += dt;

        if (unifiedState === "Productive") {
            m.current_streak += dt;
            if (m.current_streak > m.longest_streak) m.longest_streak = m.current_streak;
        } else {
            m.current_streak = 0;
        }
    }

    // 0-100 "focus quality": how much active time was productive, penalized for
    // frequent context-switching and time spent away from the screen.
    focusQuality() {
        const t = this.timers;
        const active = t.productive + t.distracted;
        if (t.total < 5) return 0;
        const productiveRatio = active > 0 ? t.productive / active : 0;
        const switchRate = this.metrics.switches / Math.max(1, t.total / 60); // per minute
        const switchPenalty = Math.min(0.35, switchRate * 0.05);
        const awayPenalty = Math.min(0.25, (this.metrics.away_seconds / Math.max(1, t.total)) * 0.5);
        return Math.round(Math.max(0, Math.min(1, productiveRatio - switchPenalty - awayPenalty)) * 100);
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
        this.metrics = this._freshMetrics();
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
            focus_quality: this.focusQuality(),
            metrics: {
                switches: this.metrics.switches,
                away_seconds: Math.floor(this.metrics.away_seconds),
                looking_away_seconds: Math.floor(this.metrics.looking_away_seconds),
                longest_streak: Math.floor(this.metrics.longest_streak),
            },
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
            camera_action: this.camera_action,
            screen_state: this.screen_state,
            screen_activity: this.screen_activity,
            screen_confidence: this.screen_confidence,
            vision_status: this.vision_status,
            unified_state: this.unified_state,
            focus_quality: this.focusQuality(),
            metrics: {
                switches: this.metrics.switches,
                away_seconds: Math.floor(this.metrics.away_seconds),
                looking_away_seconds: Math.floor(this.metrics.looking_away_seconds),
                longest_streak: Math.floor(this.metrics.longest_streak),
            },
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
        state.camera_action = state.camera_enabled ? camera.action : "Camera Off";
        state.screen_state = state.screen_enabled && screen.isSharing() ? screen.state : "Screen Off";
        state.screen_activity = state.screen_enabled && screen.isSharing() ? screen.activity : "Off";
        state.screen_confidence = screen.confidence || 0;
        state.vision_status = screen.visionStatus;
        const detected = getUnifiedState(state.camera_state, state.screen_state);
        state.unified_state = state.effectiveState(detected);

        if (state.unified_state === "Distracted") state.timers.distracted += dt;
        else if (state.unified_state === "Productive") state.timers.productive += dt;
        else state.timers.neutral += dt;
        state.timers.total += dt;

        state.updateMetrics(dt, state.unified_state, state.camera_action);
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
