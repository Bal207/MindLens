/* ============================================================================
   MindLens — dashboard UI layer

   A thin renderer over the in-browser engine (engine.js). The engine emits a
   status object with the same shape the original Flask app produced, so this
   file is mostly the original rendering logic with the network layer removed
   and the buttons rewired to the engine.
   ========================================================================== */

import { createEngine } from "./engine.js";

const localState = {
    isRunning: false,
    cameraEnabled: true,
    screenEnabled: true,
    customProductive: [],
    customDistracted: [],
    latestStatus: null,
    currentUnifiedState: "Neutral",
    distractedSince: null,
    lastNotifAt: null,
    notifDismissTimeout: null,
};

const DISTRACTION_THRESHOLD_MS = 15 * 1000;
const NOTIF_COOLDOWN_MS = 10 * 60 * 1000;

let engine = null;
let pendingFrame = false;

/* ---------- helpers ---------- */
function fmt(h, m, s) { return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":"); }
function toTotalSeconds(t) { return t ? (t.h || 0) * 3600 + (t.m || 0) * 60 + (t.s || 0) : 0; }
function pct(value, total) { return total > 0 ? Math.round((value / total) * 100) : 0; }
function setText(id, value) { const el = document.getElementById(id); if (el) el.textContent = value; }

/* ---------- overrides (engine is the source of truth) ---------- */
function setOverride(state) { engine.setOverride(state); }
function clearOverride() { engine.clearOverride(); }

function updateOverrideUI(data) {
    const badge = document.getElementById("override-badge");
    const clearBtn = document.getElementById("btn-clear-override");
    const active = data.override_state;

    if (active) {
        badge.style.display = "inline-flex";
        clearBtn.style.display = "inline-flex";
        badge.className = "override-badge override-badge--" + active.toLowerCase();
        const total = data.override_remaining || 0;
        const m = Math.floor(total / 60), s = total % 60;
        setText("override-countdown", `${m}:${String(s).padStart(2, "0")}`);
    } else {
        badge.style.display = "none";
        clearBtn.style.display = "none";
    }
    document.querySelectorAll(".override-btn").forEach((btn) => {
        btn.classList.toggle("override-btn--active", btn.textContent.trim() === active);
    });
}

/* ---------- distraction notifications ---------- */
function startDistractionWatcher() {
    setInterval(() => {
        if (!localState.isRunning) { localState.distractedSince = null; return; }
        const isDistracted = (localState.currentUnifiedState || "").toLowerCase() === "distracted";
        if (isDistracted) {
            if (!localState.distractedSince) localState.distractedSince = Date.now();
            const distractedForMs = Date.now() - localState.distractedSince;
            const cooldownOk = !localState.lastNotifAt || Date.now() - localState.lastNotifAt >= NOTIF_COOLDOWN_MS;
            if (distractedForMs >= DISTRACTION_THRESHOLD_MS && cooldownOk) {
                showNotif();
                localState.lastNotifAt = Date.now();
            }
        } else {
            localState.distractedSince = null;
        }
    }, 1000);
}

function showNotif() {
    const toast = document.getElementById("notif-toast");
    if (!toast) return;
    toast.classList.remove("notif-toast--visible");
    void toast.offsetWidth;
    toast.classList.add("notif-toast--visible");
    if (localState.notifDismissTimeout) clearTimeout(localState.notifDismissTimeout);
    localState.notifDismissTimeout = setTimeout(dismissNotif, 8000);
    if (typeof Notification !== "undefined" && Notification.permission === "granted") {
        new Notification("MindLens — Distraction alert", {
            body: "You've been distracted for over 15 seconds. Time to refocus!",
        });
    }
}

function dismissNotif() {
    const toast = document.getElementById("notif-toast");
    if (toast) toast.classList.remove("notif-toast--visible");
    if (localState.notifDismissTimeout) { clearTimeout(localState.notifDismissTimeout); localState.notifDismissTimeout = null; }
}

function requestNotifPermission() {
    if (typeof Notification !== "undefined" && Notification.permission === "default") Notification.requestPermission();
}

/* ---------- main render ---------- */
function updateUI(data) {
    localState.latestStatus = data;
    localState.isRunning = data.is_running;
    localState.cameraEnabled = data.camera_enabled;
    localState.screenEnabled = data.screen_enabled;
    localState.currentUnifiedState = data.unified_state || "Neutral";

    setText("timer-productive", fmt(data.timers.productive.h, data.timers.productive.m, data.timers.productive.s));
    setText("timer-distracted", fmt(data.timers.distracted.h, data.timers.distracted.m, data.timers.distracted.s));
    setText("timer-neutral", fmt(data.timers.neutral.h, data.timers.neutral.m, data.timers.neutral.s));
    setText("timer-total", fmt(data.timers.total.h, data.timers.total.m, data.timers.total.s));

    const productiveSec = toTotalSeconds(data.timers.productive);
    const distractedSec = toTotalSeconds(data.timers.distracted);
    const neutralSec = toTotalSeconds(data.timers.neutral);
    const totalSec = toTotalSeconds(data.timers.total);
    const safeTotal = totalSec || 1;

    document.getElementById("bar-productive").style.width = `${(productiveSec / safeTotal) * 100}%`;
    document.getElementById("bar-distracted").style.width = `${(distractedSec / safeTotal) * 100}%`;
    document.getElementById("bar-neutral").style.width = `${(neutralSec / safeTotal) * 100}%`;

    const activeTotal = productiveSec + distractedSec;
    const focusScorePct = activeTotal > 0 ? pct(productiveSec, activeTotal) : 0;

    const donut = document.getElementById("focus-donut");
    const productiveDeg = pct(productiveSec, totalSec) * 3.6;
    const distractedDeg = productiveDeg + pct(distractedSec, totalSec) * 3.6;
    donut.style.background = `conic-gradient(var(--green) 0deg ${productiveDeg}deg, var(--red) ${productiveDeg}deg ${distractedDeg}deg, var(--neutral) ${distractedDeg}deg 360deg)`;

    setText("donut-value", `${focusScorePct}%`);
    setText("focus-score", `${focusScorePct}%`);
    setText("saved-count", String(data.history_count || 0));

    const displayState = data.unified_state || "Neutral";
    const badge = document.getElementById("status-badge");
    const sessionBtn = document.getElementById("session-btn");
    badge.className = "status-badge";

    if (!data.is_running) {
        setText("status-label", "Stopped");
        setText("session-started", "No active session");
        sessionBtn.textContent = "Start Session";
        sessionBtn.className = "btn-session start";
    } else {
        badge.classList.add(displayState.toLowerCase());
        setText("status-label", displayState);
        sessionBtn.textContent = "Stop Session";
        sessionBtn.className = "btn-session stop";
        setText("session-started", startedLabel(data.session_started_at));
    }

    // HUD shows the granular action/activity, not just the coarse state.
    setText("cam-state-text", data.is_running && data.camera_enabled ? (data.camera_action || data.camera_state || "Idle") : "Off");
    setText("scr-state-text", screenHudLabel(data));

    document.getElementById("camera-toggle").checked = data.camera_enabled;
    document.getElementById("screen-toggle").checked = data.screen_enabled;
    applyCameraView(data);
    updateScreenShareUI(data);
    updateVisionStatus(data);
    updateFocusPanel(data);
    updateOverrideUI(data);

    ["productive", "distracted", "neutral"].forEach((type) => {
        const card = document.querySelector(`.${type}-card`);
        if (!card) return;
        card.classList.remove("active-productive", "active-distracted", "active-neutral");
        if (data.is_running && displayState.toLowerCase() === type) card.classList.add(`active-${type}`);
    });
}

function startedLabel(isoValue) {
    if (!isoValue) return "Active session";
    const date = new Date(isoValue);
    if (Number.isNaN(date.getTime())) return "Active session";
    return `Started ${date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
}

function scheduleUpdate(data) {
    localState.latestStatus = data;
    if (pendingFrame) return;
    pendingFrame = true;
    requestAnimationFrame(() => { pendingFrame = false; if (localState.latestStatus) updateUI(localState.latestStatus); });
}

/* ---------- camera view ---------- */
function applyCameraView(data) {
    const live = document.getElementById("camera-live");
    const overlay = document.getElementById("camera-offline");
    const hud = document.getElementById("feed-hud");
    const title = document.getElementById("offline-title");
    const sub = document.getElementById("offline-sub");

    let showLive = false, heading = "Camera off", subtext = "Start a session to see your live feed", loading = false;

    if (!data.is_running) {
        // defaults
    } else if (!data.camera_enabled) {
        heading = "Camera detection off";
        subtext = "Turn on camera detection to see your live feed";
    } else {
        const st = data.camera_state || "";
        if (st === "Initializing") { heading = "Starting camera…"; subtext = "Warming up the lens"; loading = true; }
        else if (["Camera Unavailable", "Camera Off", "Error"].includes(st)) { heading = "Camera unavailable"; subtext = "We couldn't access a camera — check browser permissions"; }
        else showLive = true;
    }

    if (showLive) {
        live.style.display = "block";
        overlay.style.display = "none";
        hud.style.display = "flex";
    } else {
        live.style.display = "none";
        overlay.style.display = "flex";
        overlay.classList.toggle("is-loading", loading);
        hud.style.display = "none";
        title.textContent = heading;
        sub.textContent = subtext;
    }
}

function screenHudLabel(data) {
    if (!data.is_running || !data.screen_enabled) return "Off";
    if (!engine || !engine.isScreenSharing()) return "Not shared";
    const activity = data.screen_activity && data.screen_activity !== "Off" ? data.screen_activity : (data.screen_state || "Neutral");
    if (data.vision_status === "loading") return `${activity} · AI loading…`;
    return activity;
}

function updateVisionStatus(data) {
    const el = document.getElementById("vision-status");
    if (!el) return;
    if (!data.is_running || !data.screen_enabled || !engine.isScreenSharing()) { el.style.display = "none"; return; }
    el.style.display = "block";
    const map = {
        loading: "◐ Loading on-device vision model (one-time download)…",
        ready: "● Vision AI active · fully on-device",
        failed: "○ Vision AI unavailable — using OCR only",
        off: "",
    };
    el.textContent = map[data.vision_status] || "";
    el.className = "vision-status vision-" + (data.vision_status || "off");
}

function updateFocusPanel(data) {
    const q = data.focus_quality || 0;
    setText("stat-quality", `${q}%`);
    const m = data.metrics || {};
    setText("stat-switches", String(m.switches || 0));
    setText("stat-streak", formatDuration(m.longest_streak || 0));
    setText("stat-away", formatDuration((m.away_seconds || 0) + (m.looking_away_seconds || 0)));
    const label = data.is_running
        ? (q >= 75 ? "Deep focus" : q >= 45 ? "Steady" : q > 0 ? "Scattered" : "Warming up")
        : "—";
    setText("focus-quality-label", label);
}

function updateScreenShareUI(data) {
    const btn = document.getElementById("share-screen-btn");
    if (!btn) return;
    const sharing = engine && engine.isScreenSharing();
    btn.style.display = data.is_running && data.screen_enabled ? "inline-flex" : "none";
    btn.textContent = sharing ? "Stop sharing screen" : "Share your screen";
    btn.classList.toggle("is-sharing", !!sharing);
}

/* ---------- controls ---------- */
async function toggleSession() {
    if (!localState.isRunning) {
        requestNotifPermission();
        await engine.startSession();
    } else {
        engine.stopSession();
        localState.distractedSince = null;
        renderHistory(engine.state.getHistory());
    }
}

function toggleCamera() {
    const on = document.getElementById("camera-toggle").checked;
    engine.enableCamera(on);
}

function toggleScreen() {
    const on = document.getElementById("screen-toggle").checked;
    engine.enableScreen(on);
}

async function shareScreen() {
    if (engine.isScreenSharing()) { engine.stopScreen(); return; }
    const ok = await engine.shareScreen();
    if (!ok) {
        const sub = document.getElementById("scr-state-text");
        if (sub) sub.textContent = "Screen share blocked";
    }
}

/* ---------- custom labels ---------- */
function syncLabels() {
    localStorage.setItem("mindlens-labels", JSON.stringify({
        productive: localState.customProductive,
        distracted: localState.customDistracted,
    }));
    if (engine) engine.setLabels(localState.customProductive, localState.customDistracted);
}

function loadLocalLabels() {
    try {
        const stored = JSON.parse(localStorage.getItem("mindlens-labels") || "{}");
        localState.customProductive = Array.isArray(stored.productive) ? stored.productive : [];
        localState.customDistracted = Array.isArray(stored.distracted) ? stored.distracted : [];
    } catch (_) {
        localState.customProductive = [];
        localState.customDistracted = [];
    }
}

function addLabel(type) {
    const input = document.getElementById(`${type}-input`);
    const val = input.value.trim();
    if (!val) return;
    const arr = type === "productive" ? localState.customProductive : localState.customDistracted;
    if (!arr.some((item) => item.toLowerCase() === val.toLowerCase())) { arr.push(val); renderChips(); syncLabels(); }
    input.value = "";
}

function removeLabel(type, val) {
    if (type === "productive") localState.customProductive = localState.customProductive.filter((v) => v !== val);
    else localState.customDistracted = localState.customDistracted.filter((v) => v !== val);
    renderChips();
    syncLabels();
}

function renderChips() {
    ["productive", "distracted"].forEach((type) => {
        const container = document.getElementById(`${type}-chips`);
        const items = type === "productive" ? localState.customProductive : localState.customDistracted;
        container.replaceChildren(...items.map((val) => {
            const chip = document.createElement("div");
            chip.className = `chip ${type}`;
            chip.append(document.createTextNode(val));
            const btn = document.createElement("button");
            btn.className = "chip-remove";
            btn.type = "button";
            btn.textContent = "x";
            btn.setAttribute("aria-label", `Remove ${val}`);
            btn.addEventListener("click", () => removeLabel(type, val));
            chip.append(btn);
            return chip;
        }));
    });
}

/* ---------- history ---------- */
function renderHistory(sessions) {
    const list = document.getElementById("history-list");
    const chart = document.getElementById("history-chart");
    setText("saved-count", String(sessions.length));

    if (!sessions.length) {
        chart.innerHTML = "";
        list.innerHTML = '<div class="empty-state">No saved sessions yet.</div>';
        return;
    }

    const newest = sessions.slice(0, 8);
    const maxScore = Math.max(...newest.map(sessionFocusScore), 1);
    chart.replaceChildren(...newest.slice().reverse().map((session) => {
        const bar = document.createElement("div");
        const score = sessionFocusScore(session);
        bar.className = "history-bar";
        bar.style.height = `${Math.max(8, (score / maxScore) * 100)}%`;
        bar.title = `${score}% productive`;
        return bar;
    }));

    list.replaceChildren(...sessions.slice(0, 6).map((session) => {
        const row = document.createElement("div");
        row.className = "history-row";
        const started = session.started_at ? new Date(session.started_at) : null;
        const label = started && !Number.isNaN(started.getTime())
            ? started.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
            : "Saved session";
        const total = session.seconds?.total || toTotalSeconds(session.timers?.total);
        const details = document.createElement("div");
        details.innerHTML = `<div class="history-title">${label}</div><div class="history-meta">${formatDuration(total)} total</div>`;
        const score = document.createElement("div");
        score.className = "history-score";
        score.textContent = `${sessionFocusScore(session)}%`;
        row.append(details, score);
        return row;
    }));
}

function sessionFocusScore(session) {
    const productive = session.seconds?.productive || toTotalSeconds(session.timers?.productive);
    const distracted = session.seconds?.distracted || toTotalSeconds(session.timers?.distracted);
    const activeTotal = productive + distracted;
    return activeTotal > 0 ? pct(productive, activeTotal) : 0;
}

function formatDuration(totalSeconds) {
    const total = Math.max(0, Math.round(totalSeconds || 0));
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
}

/* ---------- boot ---------- */
function onEngineError(kind, err) {
    console.warn(`[MindLens] ${kind} error:`, err);
    if (kind === "screen") {
        const el = document.getElementById("scr-state-text");
        if (el) el.textContent = "Screen share declined";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadLocalLabels();
    renderChips();

    engine = createEngine({
        video: document.getElementById("camera-video"),
        overlay: document.getElementById("camera-overlay"),
        onStatus: scheduleUpdate,
        onError: onEngineError,
    });
    engine.setLabels(localState.customProductive, localState.customDistracted);

    // Wire buttons.
    document.getElementById("session-btn").addEventListener("click", toggleSession);
    document.getElementById("camera-toggle").addEventListener("change", toggleCamera);
    document.getElementById("screen-toggle").addEventListener("change", toggleScreen);
    document.getElementById("share-screen-btn").addEventListener("click", shareScreen);
    document.getElementById("btn-clear-override").addEventListener("click", clearOverride);
    document.querySelectorAll(".override-btn").forEach((btn) => {
        btn.addEventListener("click", () => setOverride(btn.textContent.trim()));
    });
    document.querySelectorAll("[data-add]").forEach((btn) => {
        btn.addEventListener("click", () => addLabel(btn.getAttribute("data-add")));
    });
    document.querySelector(".notif-close").addEventListener("click", dismissNotif);

    ["productive", "distracted"].forEach((type) => {
        document.getElementById(`${type}-input`).addEventListener("keydown", (e) => {
            if (e.key === "Enter") addLabel(type);
        });
    });

    updateUI(engine.getStatus());
    renderHistory(engine.state.getHistory());
    startDistractionWatcher();
});
