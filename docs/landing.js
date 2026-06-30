/* ============================================================
   MindLens landing — interactions
   - scroll progress + nav blur
   - IntersectionObserver scroll-reveal
   - count-up stats + focus ring
   - OS detection -> GitHub Releases download links
   ============================================================ */

/* ------------------------------------------------------------------
   DOWNLOAD CONFIG  — set this once and the whole page wires itself up.
   Point GITHUB_REPO at "owner/repo". The buttons link to the assets of
   your LATEST GitHub release, so you never have to touch URLs again:
   just publish a release whose asset filenames match ASSETS below.
   ------------------------------------------------------------------ */
const GITHUB_REPO = "Bal207/MindLens"; // CI also auto-syncs this to the deploying repo

const ASSETS = {
    mac:     "MindLens-macOS.dmg",
    windows: "MindLens-Windows.zip",
    linux:   "MindLens-Linux.tar.gz",
};

const OS_META = {
    mac:     { label: "Download for macOS",  note: "macOS 11 Big Sur or later · Apple Silicon & Intel" },
    windows: { label: "Download for Windows", note: "Windows 10 or later · 64-bit" },
    linux:   { label: "Download for Linux",   note: "Modern 64-bit distributions" },
};

function releaseUrl(os) {
    const asset = ASSETS[os];
    if (!asset) return `https://github.com/${GITHUB_REPO}/releases/latest`;
    return `https://github.com/${GITHUB_REPO}/releases/latest/download/${asset}`;
}

function detectOS() {
    const ua = (navigator.userAgent || "").toLowerCase();
    const platform = (navigator.platform || "").toLowerCase();
    if (/mac/.test(platform) || /mac os x|macintosh/.test(ua)) return "mac";
    if (/win/.test(platform) || /windows/.test(ua)) return "windows";
    if (/linux|x11/.test(platform) || /linux/.test(ua)) return "linux";
    return "mac"; // sensible default
}

// The downloaded app isn't Apple-notarized, so macOS quarantines it on first
// launch. Show the "right-click > Open" note only when macOS is the target.
function toggleMacNote(os) {
    const note = document.getElementById("mac-note");
    if (note) note.hidden = os !== "mac";
}

function wireDownloads() {
    const os = detectOS();
    const meta = OS_META[os];

    toggleMacNote(os);

    const macCopy = document.getElementById("mac-copy");
    if (macCopy) {
        macCopy.addEventListener("click", async () => {
            const cmd = document.getElementById("mac-cmd").textContent.trim();
            try {
                await navigator.clipboard.writeText(cmd);
                const prev = macCopy.textContent;
                macCopy.textContent = "Copied";
                setTimeout(() => { macCopy.textContent = prev; }, 1500);
            } catch (_) { /* clipboard blocked; user can select manually */ }
        });
    }

    // Primary big button
    const primary = document.getElementById("primary-download");
    if (primary) {
        primary.href = releaseUrl(os);
        document.getElementById("primary-download-label").textContent = meta.label;
        document.getElementById("primary-download-note").textContent = meta.note;
    }

    // Hero button label
    const heroOs = document.getElementById("hero-download-os");
    if (heroOs) heroOs.textContent = { mac: "macOS", windows: "Windows", linux: "Linux" }[os];

    // Per-OS small links
    document.querySelectorAll("#os-links a[data-os]").forEach((a) => {
        const target = a.getAttribute("data-os");
        a.href = releaseUrl(target);
        if (target === os) a.classList.add("active");
        a.addEventListener("click", () => {
            // Update the primary button to whatever the user explicitly chose.
            if (primary) {
                primary.href = releaseUrl(target);
                document.getElementById("primary-download-label").textContent = OS_META[target].label;
                document.getElementById("primary-download-note").textContent = OS_META[target].note;
            }
            document.querySelectorAll("#os-links a").forEach((x) => x.classList.remove("active"));
            a.classList.add("active");
            toggleMacNote(target);
        });
    });
}

/* ---------- scroll progress + nav ---------- */
function onScroll() {
    const h = document.documentElement;
    const scrolled = h.scrollTop / (h.scrollHeight - h.clientHeight);
    const bar = document.getElementById("scroll-progress");
    if (bar) bar.style.width = `${Math.min(1, Math.max(0, scrolled)) * 100}%`;

    const nav = document.getElementById("nav");
    if (nav) nav.classList.toggle("scrolled", h.scrollTop > 24);
}

/* ---------- count up ---------- */
function animateCount(el) {
    const target = parseFloat(el.getAttribute("data-count"));
    const suffix = el.getAttribute("data-suffix") || "";
    const dur = 1100;
    const start = performance.now();
    function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(target * eased) + suffix;
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

/* ---------- focus ring ---------- */
function animateRing() {
    const fill = document.getElementById("ring-fill");
    const num = document.getElementById("ring-num");
    if (!fill || !num) return;
    const pct = 84;
    const circumference = 2 * Math.PI * 52; // ~327
    fill.style.strokeDashoffset = String(circumference * (1 - pct / 100));
    const start = performance.now();
    function tick(now) {
        const t = Math.min(1, (now - start) / 1600);
        const eased = 1 - Math.pow(1 - t, 3);
        num.textContent = Math.round(pct * eased);
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

/* ---------- reveal observer ---------- */
function initReveal() {
    const els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) {
        els.forEach((el) => el.classList.add("in"));
        return;
    }
    const io = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            el.classList.add("in");

            el.querySelectorAll("[data-count]").forEach(animateCount);
            if (el.querySelector("#ring") || el.id === "hero-device") animateRing();

            io.unobserve(el);
        });
    }, { threshold: 0.18, rootMargin: "0px 0px -8% 0px" });

    els.forEach((el) => io.observe(el));
}

document.addEventListener("DOMContentLoaded", () => {
    wireDownloads();
    initReveal();
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    // The hero device is above-the-fold; reveal + ring it immediately.
    requestAnimationFrame(() => {
        document.querySelectorAll(".hero .reveal").forEach((el) => el.classList.add("in"));
        animateRing();
    });
});
