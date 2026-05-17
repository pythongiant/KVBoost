/* ─────────────────────────────────────────────────────────────────────
   KVBoost site — interactive bits.
   Loads manifest.json and wires up the explorer, search, theme,
   hero stats, copy-to-clipboard, and feature/group navigation.
   No framework, no build step.
   ─────────────────────────────────────────────────────────────────────*/

const GITHUB_BASE = "https://github.com/pythongiant/kvboost/blob/main";

const state = {
    manifest: null,
    activeGroup: "all",
    query: "",
    selected: null,
};

document.addEventListener("DOMContentLoaded", () => {
    wireThemeToggle();
    wireCopyButtons();
    loadManifest();
    renderHeroStats();
    renderFeatureGrid();
    renderGroupFilter();
    renderModuleList();
    wireSearch();
    wireRandomPick();
    wireFeatureJump();
    renderGitInfo();
    // Highlight any code blocks that were rendered statically.
    if (window.hljs) window.hljs.highlightAll();
});

/* ──────────────────────────────────────── Manifest load ───────────────*/

function loadManifest() {
    // The manifest is shipped as a regular <script> tag (manifest.js) that
    // sets window.MANIFEST — works under file://, http://, and HTTPS alike.
    if (window.MANIFEST) {
        state.manifest = window.MANIFEST;
        return;
    }
    document.querySelector("#module-list").innerHTML =
        `<li style="color:#ec4899">manifest.js not found. Run <code>python docs/site/generate_manifest.py</code>.</li>`;
}

/* ──────────────────────────────────────── Theme ───────────────────────*/

function wireThemeToggle() {
    const btn = document.querySelector("#theme-toggle");
    const stored = localStorage.getItem("kvboost-theme");
    if (stored) document.documentElement.dataset.theme = stored;

    btn.addEventListener("click", () => {
        const cur = document.documentElement.dataset.theme === "dark" ? "dark" : matchesDark() ? "dark" : "light";
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("kvboost-theme", next);
    });
}

function matchesDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

/* ──────────────────────────────────────── Copy buttons ────────────────*/

function wireCopyButtons() {
    document.querySelectorAll("[data-copy]").forEach(btn => {
        btn.addEventListener("click", () => {
            const sel = btn.getAttribute("data-copy");
            const node = document.querySelector(sel);
            if (!node) return;
            const text = node.innerText.trim();
            navigator.clipboard.writeText(text).then(() => toast("copied"));
        });
    });
}

function toast(msg) {
    let t = document.querySelector(".toast");
    if (!t) {
        t = document.createElement("div");
        t.className = "toast";
        document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(t._h);
    t._h = setTimeout(() => t.classList.remove("show"), 1400);
}

/* ──────────────────────────────────────── Hero stats ──────────────────*/

function renderHeroStats() {
    if (!state.manifest) return;
    const s = state.manifest.stats;
    const row = document.querySelector("#hero-stats");
    const stats = [
        { num: s.module_count,  label: "Python modules" },
        { num: s.total_loc.toLocaleString(),  label: "Lines of code" },
        { num: s.feature_count, label: "Feature areas" },
        { num: s.extra_file_count, label: "Native files (Rust / CUDA)" },
    ];
    row.innerHTML = stats.map(s => `
        <div class="stat">
            <span class="num" data-counter="${s.num}">${s.num}</span>
            <span class="label">${escapeHtml(s.label)}</span>
        </div>`).join("");
    animateCounters();
}

function animateCounters() {
    document.querySelectorAll("[data-counter]").forEach(el => {
        const target = el.getAttribute("data-counter").replace(/,/g, "");
        const n = parseInt(target, 10);
        if (Number.isNaN(n)) return;
        const dur = 800;
        const t0 = performance.now();
        const tick = (now) => {
            const p = Math.min(1, (now - t0) / dur);
            const eased = 1 - Math.pow(1 - p, 3);
            el.textContent = Math.round(n * eased).toLocaleString();
            if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    });
}

/* ──────────────────────────────────────── Feature grid ────────────────*/

function renderFeatureGrid() {
    if (!state.manifest) return;
    const grid = document.querySelector("#feature-grid");
    const counts = {};
    for (const m of state.manifest.modules) {
        counts[m.group] = (counts[m.group] || 0) + 1;
    }

    grid.innerHTML = state.manifest.feature_groups.map(g => `
        <div class="feature-card" data-jump-group="${g.id}">
            <span class="stripe" style="background: ${g.color}"></span>
            <span class="emoji">${g.emoji}</span>
            <h3>${escapeHtml(g.label)}</h3>
            <p>${escapeHtml(g.tagline)}</p>
            <div class="module-count">${counts[g.id] || 0} module${(counts[g.id] || 0) === 1 ? "" : "s"}</div>
        </div>`).join("");

    grid.querySelectorAll(".feature-card").forEach(card => {
        card.addEventListener("click", () => jumpToGroup(card.dataset.jumpGroup));
    });
}

/* ──────────────────────────────────────── Group filter chips ──────────*/

function renderGroupFilter() {
    if (!state.manifest) return;
    const wrap = document.querySelector("#group-filter");
    const groups = [{ id: "all", label: "all", color: "var(--fg-muted)", emoji: "✨" }, ...state.manifest.feature_groups];
    wrap.innerHTML = groups.map(g => `
        <button class="chip ${g.id === state.activeGroup ? "active" : ""}" data-group="${g.id}">
            ${g.emoji ? g.emoji + " " : ""}${escapeHtml(g.label)}
        </button>`).join("");
    wrap.querySelectorAll(".chip").forEach(c => {
        c.addEventListener("click", () => {
            state.activeGroup = c.dataset.group;
            renderGroupFilter();
            renderModuleList();
        });
    });
}

/* ──────────────────────────────────────── Module list ─────────────────*/

function renderModuleList() {
    if (!state.manifest) return;
    const ul = document.querySelector("#module-list");
    const q = state.query.toLowerCase();
    const filtered = state.manifest.modules.filter(m => {
        if (state.activeGroup !== "all" && m.group !== state.activeGroup) return false;
        if (!q) return true;
        if (m.name.toLowerCase().includes(q)) return true;
        if (m.summary.toLowerCase().includes(q)) return true;
        if (m.classes.some(s => s.name.toLowerCase().includes(q))) return true;
        if (m.functions.some(s => s.name.toLowerCase().includes(q))) return true;
        return false;
    });

    if (filtered.length === 0) {
        ul.innerHTML = `<li style="color: var(--fg-muted)">no matches</li>`;
        return;
    }

    ul.innerHTML = filtered.map(m => `
        <li data-mod="${escapeAttr(m.name)}" class="${state.selected === m.name ? "selected" : ""}">
            <span class="dot-color" style="background: ${m.color}"></span>
            <span class="mod-name" title="${escapeAttr(m.name)}">${escapeHtml(shortName(m.name))}</span>
            <span class="mod-loc">${m.loc}</span>
        </li>`).join("");

    ul.querySelectorAll("li[data-mod]").forEach(li => {
        li.addEventListener("click", () => selectModule(li.dataset.mod));
    });
}

function shortName(dotted) {
    const parts = dotted.split(".");
    if (parts.length <= 2) return dotted;
    return parts.slice(1).join(".");  // strip leading "kvboost"
}

/* ──────────────────────────────────────── Module detail ───────────────*/

function selectModule(name) {
    state.selected = name;
    const m = state.manifest.modules.find(x => x.name === name);
    if (!m) return;
    renderModuleList();  // re-render to update .selected
    renderModuleDetail(m);
}

function renderModuleDetail(m) {
    const wrap = document.querySelector("#module-detail");
    const group = state.manifest.feature_groups.find(g => g.id === m.group);
    const githubLink = `${GITHUB_BASE}/${m.rel_path}`;

    const symbolsHtml = (list, kind) => list.length === 0 ? "" : `
        <h4>${kind === "class" ? "Classes" : "Functions"} (${list.length})</h4>
        <div class="symbol-list">
            ${list.map(s => `
                <div class="sym ${s.is_public ? "" : "private"}">
                    <span class="sym-kind ${kind}">${kind}</span>
                    <a class="sym-name" href="${githubLink}#L${s.lineno}" target="_blank" rel="noopener">${escapeHtml(s.name)}</a>
                    <span class="sym-summary">${escapeHtml(s.summary || "—")}</span>
                </div>`).join("")}
        </div>`;

    wrap.innerHTML = `
        <div class="header">
            <span class="emoji">${group ? group.emoji : "📦"}</span>
            <span class="name">${escapeHtml(m.name)}</span>
            <span class="group-pill" style="background: ${m.color}22; color: ${m.color}">${group ? group.label : m.group}</span>
        </div>
        <div class="meta-row">
            <span>${m.loc} loc</span>
            <span>${(m.bytes / 1024).toFixed(1)} KB</span>
            <a href="${githubLink}" target="_blank" rel="noopener">view on GitHub →</a>
        </div>
        <p class="summary">${escapeHtml(m.summary) || "<em>(no docstring summary)</em>"}</p>
        ${m.docstring ? `<div class="docstring">${escapeHtml(m.docstring)}</div>` : ""}
        ${symbolsHtml(m.classes,   "class")}
        ${symbolsHtml(m.functions, "function")}
    `;
}

/* ──────────────────────────────────────── Search ──────────────────────*/

function wireSearch() {
    const input = document.querySelector("#module-search");
    input.addEventListener("input", () => {
        state.query = input.value.trim();
        renderModuleList();
    });
}

/* ──────────────────────────────────────── Random pick ─────────────────*/

function wireRandomPick() {
    const btn = document.querySelector("#random-pick");
    if (!btn) return;
    btn.addEventListener("click", () => {
        if (!state.manifest) return;
        const candidates = state.manifest.modules.filter(m =>
            state.activeGroup === "all" || m.group === state.activeGroup
        );
        if (candidates.length === 0) return;
        const m = candidates[Math.floor(Math.random() * candidates.length)];
        selectModule(m.name);
    });
}

/* ──────────────────────────────────────── Feature jumps ───────────────*/

function wireFeatureJump() {
    document.querySelectorAll("[data-jump-group]").forEach(el => {
        el.addEventListener("click", (e) => {
            e.preventDefault();
            jumpToGroup(el.dataset.jumpGroup);
        });
    });
}

function jumpToGroup(groupId) {
    state.activeGroup = groupId;
    state.query = "";
    const input = document.querySelector("#module-search");
    if (input) input.value = "";
    renderGroupFilter();
    renderModuleList();
    document.querySelector("#explore").scrollIntoView({ behavior: "smooth" });
    // Auto-select the first module in the group for instant feedback.
    if (state.manifest) {
        const first = state.manifest.modules.find(m =>
            groupId === "all" || m.group === groupId
        );
        if (first) selectModule(first.name);
    }
}

/* ──────────────────────────────────────── Git info ────────────────────*/

function renderGitInfo() {
    if (!state.manifest) return;
    const g = state.manifest.git;
    const el = document.querySelector("#git-info");
    if (!el) return;
    const bits = [];
    if (g.branch) bits.push(g.branch);
    if (g.commit) bits.push(g.commit);
    el.textContent = bits.length ? `git: ${bits.join(" / ")}` : "";
}

/* ──────────────────────────────────────── Helpers ─────────────────────*/

function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
function escapeAttr(s) { return escapeHtml(s); }
