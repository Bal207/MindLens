/* ============================================================
   MindLens landing — interactions
   - scroll progress + nav blur
   - IntersectionObserver scroll-reveal
   - count-up stats + focus ring
   - wire the "Launch" buttons to the in-browser app
   ============================================================ */

/* ------------------------------------------------------------------
   LAUNCH CONFIG — MindLens now runs entirely in the browser. The app is
   deployed alongside this landing page at ./app/, so every launch button
   just points there. There are no downloads to host anymore.
   ------------------------------------------------------------------ */
const APP_URL = "app/";

function wireLaunch() {
    document.querySelectorAll("[data-launch]").forEach((el) => { el.href = APP_URL; });
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
    wireLaunch();
    initReveal();
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    // The hero device is above-the-fold; reveal + ring it immediately.
    requestAnimationFrame(() => {
        document.querySelectorAll(".hero .reveal").forEach((el) => el.classList.add("in"));
        animateRing();
    });
});
