const localState = {
    isRunning: false,
    cameraEnabled: true,
    screenEnabled: true,
    customProductive: [],
    customDistracted: [],
    latestStatus: null,
};

let eventSource = null;
let pendingFrame = false;

function fmt(h, m, s) {
    return [h, m, s].map(v => String(v).padStart(2, '0')).join(':');
}

function toTotalSeconds(t) {
    if (!t) return 0;
    return (t.h || 0) * 3600 + (t.m || 0) * 60 + (t.s || 0);
}

function pct(value, total) {
    return total > 0 ? Math.round((value / total) * 100) : 0;
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function updateUI(data) {
    localState.latestStatus = data;
    localState.isRunning = data.is_running;
    localState.cameraEnabled = data.camera_enabled;
    localState.screenEnabled = data.screen_enabled;

    setText('timer-productive', fmt(data.timers.productive.h, data.timers.productive.m, data.timers.productive.s));
    setText('timer-distracted', fmt(data.timers.distracted.h, data.timers.distracted.m, data.timers.distracted.s));
    setText('timer-neutral', fmt(data.timers.neutral.h, data.timers.neutral.m, data.timers.neutral.s));
    setText('timer-total', fmt(data.timers.total.h, data.timers.total.m, data.timers.total.s));

    const productiveSec = toTotalSeconds(data.timers.productive);
    const distractedSec = toTotalSeconds(data.timers.distracted);
    const neutralSec = toTotalSeconds(data.timers.neutral);
    const totalSec = toTotalSeconds(data.timers.total);
    const safeTotal = totalSec || 1;
    const productivePct = pct(productiveSec, totalSec);
    const distractedPct = pct(distractedSec, totalSec);
    const neutralPct = pct(neutralSec, totalSec);

    document.getElementById('bar-productive').style.width = `${productiveSec / safeTotal * 100}%`;
    document.getElementById('bar-distracted').style.width = `${distractedSec / safeTotal * 100}%`;
    document.getElementById('bar-neutral').style.width = `${neutralSec / safeTotal * 100}%`;

    const activeTotal = productiveSec + distractedSec;
    const focusScorePct = activeTotal > 0 ? pct(productiveSec, activeTotal) : 0;

    const donut = document.getElementById('focus-donut');
    const productiveDeg = productivePct * 3.6;
    const distractedDeg = productiveDeg + distractedPct * 3.6;
    donut.style.background = `conic-gradient(var(--green) 0deg ${productiveDeg}deg, var(--red) ${productiveDeg}deg ${distractedDeg}deg, var(--neutral) ${distractedDeg}deg 360deg)`;
    setText('donut-value', `${focusScorePct}%`);
    setText('focus-score', `${focusScorePct}%`);
    setText('saved-count', String(data.history_count || 0));

    const badge = document.getElementById('status-badge');
    const sessionBtn = document.getElementById('session-btn');
    badge.className = 'status-badge';

    if (!data.is_running) {
        setText('status-label', 'Stopped');
        setText('session-started', 'No active session');
        sessionBtn.textContent = 'Start Session';
        sessionBtn.className = 'btn-session start';
    } else {
        const currentState = (data.unified_state || 'neutral').toLowerCase();
        badge.classList.add(currentState);
        setText('status-label', data.unified_state || 'Neutral');
        sessionBtn.textContent = 'Stop Session';
        sessionBtn.className = 'btn-session stop';
        setText('session-started', startedLabel(data.session_started_at));
    }

    const camState = data.camera_state || 'Idle';
    const scrState = data.screen_state || 'Neutral';
    setText('cam-state-text', camState);
    setText('scr-state-text', scrState);

    const camBadge = document.getElementById('cam-state-badge');
    camBadge.textContent = camState;
    camBadge.className = `state-indicator ${stateClass(camState)}`;

    document.getElementById('camera-toggle').checked = data.camera_enabled;
    document.getElementById('screen-toggle').checked = data.screen_enabled;
    setCameraVisibility(data.camera_enabled);

    ['productive', 'distracted', 'neutral'].forEach(type => {
        const card = document.querySelector(`.${type}-card`);
        if (!card) return;
        card.classList.remove('active-productive', 'active-distracted', 'active-neutral');
        if (data.is_running && (data.unified_state || '').toLowerCase() === type) {
            card.classList.add(`active-${type}`);
        }
    });
}

function startedLabel(isoValue) {
    if (!isoValue) return 'Active session';
    const date = new Date(isoValue);
    if (Number.isNaN(date.getTime())) return 'Active session';
    return `Started ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
}

function stateClass(s) {
    const lower = (s || '').toLowerCase();
    if (lower.includes('phone') || lower === 'distracted') return 'distracted';
    if (lower.includes('study') || lower.includes('reading') || lower === 'productive') return 'productive';
    return 'neutral';
}

function scheduleUpdate(data) {
    localState.latestStatus = data;
    if (pendingFrame) return;
    pendingFrame = true;
    requestAnimationFrame(() => {
        pendingFrame = false;
        if (localState.latestStatus) updateUI(localState.latestStatus);
    });
}

function startSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/stream');
    eventSource.onmessage = e => {
        try {
            scheduleUpdate(JSON.parse(e.data));
        } catch (_) {
            // Ignore partial stream messages.
        }
    };
    eventSource.onerror = () => {
        eventSource.close();
        setTimeout(startSSE, 2000);
    };
}

async function toggleSession() {
    const endpoint = localState.isRunning ? '/api/stop' : '/api/start';
    const res = await fetch(endpoint, { method: 'POST' });
    const data = await res.json();
    localState.isRunning = !localState.isRunning;
    if (data.session) loadHistory();
    loadStatus();
}

async function toggleCamera() {
    const res = await fetch('/api/toggle/camera', { method: 'POST' });
    const data = await res.json();
    localState.cameraEnabled = data.camera_enabled;
    setCameraVisibility(localState.cameraEnabled);
    document.getElementById('camera-toggle').checked = localState.cameraEnabled;
}

function setCameraVisibility(enabled) {
    const feed = document.getElementById('camera-feed');
    const offline = document.getElementById('camera-offline');
    if (enabled) {
        if (feed.style.display === 'none') feed.src = `/video_feed?${Date.now()}`;
        feed.style.display = 'block';
        offline.style.display = 'none';
    } else {
        feed.style.display = 'none';
        offline.style.display = 'flex';
    }
}

async function toggleScreen() {
    const res = await fetch('/api/toggle/screen', { method: 'POST' });
    const data = await res.json();
    localState.screenEnabled = data.screen_enabled;
    document.getElementById('screen-toggle').checked = localState.screenEnabled;
}

async function syncLabels() {
    localStorage.setItem('mindlens-labels', JSON.stringify({
        productive: localState.customProductive,
        distracted: localState.customDistracted,
    }));
    await fetch('/api/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            productive: localState.customProductive,
            distracted: localState.customDistracted,
        }),
    });
}

function loadLocalLabels() {
    try {
        const stored = JSON.parse(localStorage.getItem('mindlens-labels') || '{}');
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

    const arr = type === 'productive' ? localState.customProductive : localState.customDistracted;
    if (!arr.some(item => item.toLowerCase() === val.toLowerCase())) {
        arr.push(val);
        renderChips();
        syncLabels();
    }
    input.value = '';
}

function removeLabel(type, val) {
    if (type === 'productive') {
        localState.customProductive = localState.customProductive.filter(v => v !== val);
    } else {
        localState.customDistracted = localState.customDistracted.filter(v => v !== val);
    }
    renderChips();
    syncLabels();
}

function renderChips() {
    ['productive', 'distracted'].forEach(type => {
        const container = document.getElementById(`${type}-chips`);
        const items = type === 'productive' ? localState.customProductive : localState.customDistracted;
        container.replaceChildren(...items.map(val => {
            const chip = document.createElement('div');
            chip.className = `chip ${type}`;
            chip.append(document.createTextNode(val));

            const btn = document.createElement('button');
            btn.className = 'chip-remove';
            btn.type = 'button';
            btn.textContent = 'x';
            btn.setAttribute('aria-label', `Remove ${val}`);
            btn.addEventListener('click', () => removeLabel(type, val));
            chip.append(btn);
            return chip;
        }));
    });
}

async function loadStatus() {
    const res = await fetch('/api/status');
    updateUI(await res.json());
}

async function loadHistory() {
    const res = await fetch('/api/history');
    const data = await res.json();
    renderHistory(data.sessions || []);
}

function renderHistory(sessions) {
    const list = document.getElementById('history-list');
    const chart = document.getElementById('history-chart');
    setText('saved-count', String(sessions.length));

    if (!sessions.length) {
        chart.innerHTML = '';
        list.innerHTML = '<div class="empty-state">No saved sessions yet.</div>';
        return;
    }

    const newest = sessions.slice(0, 8);
    const maxScore = Math.max(...newest.map(sessionFocusScore), 1);
    chart.replaceChildren(...newest.slice().reverse().map(session => {
        const bar = document.createElement('div');
        const score = sessionFocusScore(session);
        bar.className = 'history-bar';
        bar.style.height = `${Math.max(8, (score / maxScore) * 100)}%`;
        bar.title = `${score}% productive`;
        return bar;
    }));

    list.replaceChildren(...sessions.slice(0, 6).map(session => {
        const row = document.createElement('div');
        row.className = 'history-row';

        const started = session.started_at ? new Date(session.started_at) : null;
        const label = started && !Number.isNaN(started.getTime())
            ? started.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
            : 'Saved session';

        const total = session.seconds?.total || toTotalSeconds(session.timers?.total);
        const details = document.createElement('div');
        details.innerHTML = `<div class="history-title">${label}</div><div class="history-meta">${formatDuration(total)} total</div>`;

        const score = document.createElement('div');
        score.className = 'history-score';
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
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m ${s}s`;
    return `${s}s`;
}

document.addEventListener('DOMContentLoaded', () => {
    loadLocalLabels();
    renderChips();
    syncLabels();
    loadStatus();
    loadHistory();
    startSSE();

    ['productive', 'distracted'].forEach(type => {
        document.getElementById(`${type}-input`).addEventListener('keydown', e => {
            if (e.key === 'Enter') addLabel(type);
        });
    });
});
