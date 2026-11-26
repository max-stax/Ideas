/* Ideas — a place to dump thoughts. The app finds related thoughts and themes on
   the phone with a small language model; nothing leaves the device except the
   sync to your private GitHub repo. */
import * as C from "./core.js";

const MODEL = "Xenova/all-MiniLM-L6-v2";
const LIB = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
const CFG_KEY = "ideas.cfg", BANK_KEY = "ideas.bank", CL_KEY = "ideas.claude", DRAFT_KEY = "ideas.draft";

const readLS = k => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
const writeLS = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

let cfg = { owner: "max-stax", dataRepo: "idea-bank", token: "", tight: C.THEME_MIN, ...(readLS(CFG_KEY) || {}) };
let bank = readLS(BANK_KEY) || { thoughts: [], deleted: {} };
let claude = readLS(CL_KEY);
const ui = { view: "dump", stack: [], q: "", sync: "", model: "off", pct: 0 };

const vecs = new Map();     // id -> Float32Array
const vecHash = new Map();  // id -> hash of the text the vector was made from
let model = null;

const $ = s => document.querySelector(s);
const main = $("#main");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const withTags = s => String(s ?? "").split(/(#[\p{L}\p{N}_-]+)/u)
  .map((part, i) => i % 2 ? `<span class="tag" data-act="tagq" data-v="${esc(part.toLowerCase())}">${esc(part)}</span>` : esc(part)).join("");
const byId = id => bank.thoughts.find(t => t.id === id);

function when(iso) {
  if (!iso) return "";
  const d = new Date(iso), now = new Date();
  const days = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / 864e5);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return d.toLocaleDateString(undefined, { weekday: "long" });
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", ...(d.getFullYear() !== now.getFullYear() ? { year: "numeric" } : {}) });
}
function toast(msg) {
  const el = $("#toast"); el.textContent = msg; el.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => (el.hidden = true), 1800);
}
function status() {
  const m = ui.model === "loading" ? `AI ${ui.pct ? ui.pct + "%" : "…"}` : ui.model === "error" ? "AI off" : "";
  $("#status").textContent = [m, ui.sync].filter(Boolean).join(" · ");
}

// ------------------------------------------------------------------ storage
function saveLocal() { writeLS(BANK_KEY, bank); }
function changed() {
  saveLocal();
  clearTimeout(changed.t); changed.t = setTimeout(sync, 800);
  embedMissing();
}

const idb = (() => {
  let p;
  const open = () => p ||= new Promise((res, rej) => {
    const r = indexedDB.open("ideas", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("vecs");
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
  const tx = async (mode, fn) => { const db = await open(); return new Promise((res, rej) => {
    const t = db.transaction("vecs", mode); const out = fn(t.objectStore("vecs"));
    t.oncomplete = () => res(out?.result); t.onerror = () => rej(t.error); }); };
  return {
    all: async () => { const db = await open(); return new Promise((res, rej) => {
      const out = new Map(); const c = db.transaction("vecs").objectStore("vecs").openCursor();
      c.onsuccess = () => { const cur = c.result; if (!cur) return res(out); out.set(cur.key, cur.value); cur.continue(); };
      c.onerror = () => rej(c.error); }); },
    put: (id, val) => tx("readwrite", s => s.put(val, id)),
    del: id => tx("readwrite", s => s.delete(id)),
  };
})();

// ------------------------------------------------------------------ model
const embedText = t => t.text.replace(/#([\p{L}\p{N}_-]+)/gu, "$1").slice(0, 1200);

async function loadModel() {
  if (model || ui.model === "loading") return;
  ui.model = "loading"; ui.pct = 0; status();
  try {
    const T = await import(LIB);
    T.env.allowLocalModels = false;
    try { T.env.backends.onnx.wasm.numThreads = 1; } catch {}
    model = await T.pipeline("feature-extraction", MODEL, {
      dtype: "q8", device: "wasm",
      progress_callback: p => { if (p.status === "progress" && /onnx/.test(p.file || "")) { ui.pct = Math.round(p.progress || 0); status(); } },
    });
    ui.model = "ready"; status();
    await embedMissing();
  } catch (e) {
    console.warn("model", e); ui.model = "error"; status();
    if (ui.view !== "dump") render();
  }
}

let embedding = false;
async function embedMissing() {
  if (!model || embedding) return;
  embedding = true;
  try {
    for (;;) {
      const todo = bank.thoughts.filter(t => vecHash.get(t.id) !== C.hash(embedText(t))).slice(0, 8);
      if (!todo.length) break;
      const out = await model(todo.map(embedText), { pooling: "mean", normalize: true });
      const rows = out.tolist();
      for (let i = 0; i < todo.length; i++) {
        const v = Float32Array.from(rows[i]), h = C.hash(embedText(todo[i]));
        vecs.set(todo[i].id, v); vecHash.set(todo[i].id, h);
        idb.put(todo[i].id, { h, v }).catch(() => {});
      }
      themeCache = null;
    }
    const live = new Set(bank.thoughts.map(t => t.id));
    for (const id of [...vecs.keys()]) if (!live.has(id)) { vecs.delete(id); vecHash.delete(id); idb.del(id).catch(() => {}); }
  } catch (e) { console.warn("embed", e); }
  embedding = false;
  refreshQuiet();
}

let themeCache = null, themeKey = "";
function autoThemes() {
  const key = cfg.tight + "|" + bank.thoughts.map(t => t.id + vecHash.get(t.id)).join(",");
  if (themeCache && key === themeKey) return themeCache;
  const items = bank.thoughts.filter(t => vecs.has(t.id));
  const groups = C.clusterIds(items.map(t => t.id), vecs, cfg.tight);
  const multi = groups.filter(g => g.length >= 2);
  const labels = C.labelThemes(multi, new Map(bank.thoughts.map(t => [t.id, t.text])));
  const themes = multi.map((g, i) => {
    const c = C.centerOf(g, vecs);
    return { ids: [c, ...g.filter(id => id !== c)], label: labels[i] };
  }).sort((a, b) => b.ids.length - a.ids.length);
  themeCache = { themes, loose: groups.filter(g => g.length === 1).length };
  themeKey = key;
  return themeCache;
}

// ------------------------------------------------------------------ sync
const b64e = s => btoa(unescape(encodeURIComponent(s)));
const b64d = s => decodeURIComponent(escape(atob(s.replace(/\n/g, ""))));
async function gh(method, path, body) {
  const r = await fetch(`https://api.github.com/repos/${cfg.owner}/${cfg.dataRepo}/contents/${path}`, {
    method, cache: "no-store", body: body ? JSON.stringify(body) : undefined,
    headers: { Authorization: `Bearer ${cfg.token}`, Accept: "application/vnd.github+json", "Content-Type": "application/json" },
  });
  if (r.status === 404 && method === "GET") return null;
  if (!r.ok) { const e = new Error(`GitHub ${r.status}`); e.status = r.status; throw e; }
  return r.json();
}
function setSync(s) { ui.sync = s; status(); }

let syncing = false, again = false;
async function sync() {
  if (!cfg.token) { setSync(""); return; }
  if (syncing) { again = true; return; }
  syncing = true; setSync("syncing…");
  try {
    for (let attempt = 0; ; attempt++) {
      const f = await gh("GET", "thoughts.json");
      const remote = f ? JSON.parse(b64d(f.content)) : { thoughts: [], deleted: {} };
      const merged = C.merge(bank, remote);
      const gotNew = !C.sameBank(merged, bank);
      bank = merged; saveLocal();
      if (gotNew) { embedMissing(); refreshQuiet(); }
      if (C.sameBank(merged, remote)) break;
      try {
        const body = { message: `ideas ${new Date().toISOString()}`, content: b64e(JSON.stringify({ updated: new Date().toISOString(), ...merged }, null, 1)) };
        if (f) body.sha = f.sha;
        await gh("PUT", "thoughts.json", body);
        break;
      } catch (e) { if ((e.status === 409 || e.status === 422) && attempt < 2) continue; throw e; }
    }
    const c = await gh("GET", "claude.json").catch(() => null);
    const next = c ? JSON.parse(b64d(c.content)) : null;
    if (JSON.stringify(next) !== JSON.stringify(claude)) { claude = next; writeLS(CL_KEY, claude); refreshQuiet(); }
    setSync("synced");
  } catch (e) {
    console.warn("sync", e);
    setSync(e.status === 401 ? "bad token" : e.status === 404 ? "repo not found" : navigator.onLine ? "sync error" : "offline");
  }
  syncing = false;
  if (again) { again = false; sync(); }
}

// ------------------------------------------------------------------ views
function thoughtRow(t) {
  return `<div class="thought" data-act="open" data-id="${t.id}">
    <div class="tx">${withTags(t.text)}</div>
    <div class="meta">${when(t.created)}</div>
  </div>`;
}

function viewDump() {
  return `<div class="capture">
      <textarea id="cap" placeholder="What's on your mind?" enterkeyhint="enter">${esc(localStorage.getItem(DRAFT_KEY) || "")}</textarea>
      <div class="bar"><span class="sp"></span><button class="btn" data-act="save" id="saveBtn">Save</button></div>
    </div>
    <input class="search" id="q" type="search" placeholder="Search" value="${esc(ui.q)}">
    <div id="dumpList"></div>`;
}
function dumpList() {
  const terms = ui.q.toLowerCase().split(/\s+/).filter(Boolean);
  const list = terms.length ? bank.thoughts.filter(t => terms.every(w => t.text.toLowerCase().includes(w))) : bank.thoughts;
  if (!list.length) return `<div class="empty">${bank.thoughts.length ? "No matches" : "Nothing yet"}</div>`;
  return `<div class="list">${list.map(t => thoughtRow(t)).join("")}</div>`;
}

// A theme's title is the first line of its most central thought.
const titleOf = text => {
  const line = (text || "").replace(/#[\p{L}\p{N}_-]+/gu, "").split("\n").map(s => s.replace(/\s+/g, " ").trim()).find(Boolean) || "";
  return line.length > 70 ? line.slice(0, 68).replace(/\s+\S*$/, "") + "…" : line;
};
const themeRow = (src, i, title, n) =>
  `<div class="trow" data-act="theme" data-src="${src}" data-i="${i}"><span class="tt">${esc(title)}</span><span class="tc">${n}</span></div>`;

function viewThemes() {
  const rows = [];
  (claude?.themes || []).forEach((th, i) => {
    const n = (th.ids || []).filter(byId).length;
    if (n) rows.push(themeRow("claude", i, th.name, n));
  });
  if (ui.model === "ready") autoThemes().themes.forEach((th, i) => rows.push(themeRow("auto", i, titleOf(byId(th.ids[0])?.text), th.ids.length)));
  if (rows.length) return `<div class="tlist">${rows.join("")}</div>`;
  if (ui.model === "error") return `<div class="empty">Couldn't load. <button class="btn ghost" data-act="retryModel">Retry</button></div>`;
  if (ui.model !== "ready") return `<div class="empty">Loading${ui.pct ? ` ${ui.pct}%` : "…"}</div>`;
  return `<div class="empty">No themes yet</div>`;
}

function themeDetail(src, i) {
  const th = src === "claude" ? claude?.themes?.[i] : autoThemes().themes[i];
  if (!th) return `<div class="empty">Gone</div>`;
  $("#title").textContent = src === "claude" ? th.name : titleOf(byId(th.ids[0])?.text);
  const items = th.ids.map(byId).filter(Boolean).sort((a, b) => (a.created || "").localeCompare(b.created || ""));
  return `<div class="list">${items.map(t => thoughtRow(t)).join("")}</div>`;
}

function thoughtDetail(id, editing) {
  const t = byId(id); if (!t) return `<div class="empty">Deleted</div>`;
  const date = new Date(t.created).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  $("#title").textContent = date;
  if (editing) return `<textarea id="edit" class="editbox" data-id="${t.id}">${esc(t.text)}</textarea>
    <div class="row"><button class="btn danger" data-act="del" data-id="${t.id}">Delete</button><span class="sp"></span>
      <button class="btn ghost" data-act="cancelEdit">Cancel</button><button class="btn" data-act="saveEdit">Save</button></div>`;
  let h = `<div class="full">${withTags(t.text)}</div>
    <div class="row"><button class="btn ghost" data-act="edit">Edit</button></div>`;
  const near = new Map();
  for (const l of claude?.links || []) if (l.a === id || l.b === id) { const o = byId(l.a === id ? l.b : l.a); if (o) near.set(o.id, o); }
  if (ui.model === "ready" && vecs.has(id)) for (const r of C.related(id, bank.thoughts, vecs, 6)) near.set(r.t.id, r.t);
  if (near.size) h += `<div class="sec">Related</div><div class="list">${[...near.values()].slice(0, 6).map(o => thoughtRow(o)).join("")}</div>`;
  return h;
}

function viewSettings() {
  return `<div class="card">
    <label class="f"><span>GitHub token</span><input type="password" id="tok" value="${esc(cfg.token)}" placeholder="github_pat_…" autocomplete="off"></label>
    <label class="f"><span>Repo</span><input type="text" id="repo" value="${esc(cfg.dataRepo)}" autocapitalize="off" autocorrect="off"></label>
    <div class="row"><button class="btn" data-act="saveTok">Save &amp; sync</button><span class="sp"></span><span class="note">${esc(ui.sync)}</span></div>
  </div>
  <div class="card">
    <label class="f"><span>Theme tightness</span><input type="range" id="tight" min="0.30" max="0.60" step="0.02" value="${cfg.tight}">
      <small style="display:flex;justify-content:space-between"><span>Fewer</span><span>More</span></small></label>
  </div>
  <div class="card"><button class="btn ghost" data-act="export">Export as Markdown</button></div>`;
}

// ------------------------------------------------------------------ render
function render() {
  const top = ui.stack[ui.stack.length - 1];
  $("#back").hidden = !top;
  document.querySelectorAll("nav.tabs button").forEach(b => b.classList.toggle("on", b.dataset.v === ui.view));
  $("#title").textContent = { dump: "Dump", themes: "Themes", settings: "Settings" }[ui.view];
  if (top?.kind === "thought") main.innerHTML = thoughtDetail(top.id, top.editing);
  else if (top?.kind === "theme") main.innerHTML = themeDetail(top.src, top.i);
  else if (ui.view === "dump") { main.innerHTML = viewDump(); $("#dumpList").innerHTML = dumpList(); bindDump(); }
  else if (ui.view === "themes") main.innerHTML = viewThemes();
  else main.innerHTML = viewSettings();
  if (ui.view === "settings" && !top) $("#tight").addEventListener("input", e => { cfg.tight = +e.target.value; writeLS(CFG_KEY, cfg); themeCache = null; });
}
// Re-render without stealing focus from something the user is typing in.
function refreshQuiet() {
  const top = ui.stack[ui.stack.length - 1];
  if (!top && ui.view === "dump") { const l = $("#dumpList"); if (l) l.innerHTML = dumpList(); return; }
  if (top?.editing || document.activeElement?.tagName === "INPUT") return;
  render();
}

function bindDump() {
  const cap = $("#cap"), q = $("#q");
  cap.addEventListener("input", () => { localStorage.setItem(DRAFT_KEY, cap.value); });
  cap.addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); saveThought(); } });
  q.addEventListener("input", () => { ui.q = q.value; $("#dumpList").innerHTML = dumpList(); });
}

function saveThought() {
  const cap = $("#cap"), text = cap.value.trim();
  if (!text) return cap.focus();
  bank.thoughts.unshift({ id: C.newId(), text, created: new Date().toISOString(), source: "phone" });
  cap.value = ""; localStorage.removeItem(DRAFT_KEY);
  $("#dumpList").innerHTML = dumpList();
  toast("Saved"); changed();
}

function exportMd() {
  const md = "# Ideas\n\n" + bank.thoughts.map(t => `- ${t.text.replace(/\n+/g, " ")}  \n  _${t.created.slice(0, 10)}${t.source === "chat" ? " · chat" : ""}_`).join("\n") + "\n";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
  a.download = `ideas-${new Date().toISOString().slice(0, 10)}.md`;
  document.body.appendChild(a); a.click(); a.remove();
}

document.addEventListener("click", async e => {
  const el = e.target.closest("[data-act]"); if (!el) return;
  const a = el.dataset.act;
  if (a === "tab") { ui.stack = []; ui.view = el.dataset.v; render(); scrollTo(0, 0); }
  else if (a === "back") { ui.stack.pop(); render(); }
  else if (a === "save") saveThought();
  else if (a === "open") { e.stopPropagation(); ui.stack.push({ kind: "thought", id: el.dataset.id }); render(); scrollTo(0, 0); }
  else if (a === "theme") { ui.stack.push({ kind: "theme", src: el.dataset.src, i: +el.dataset.i }); render(); scrollTo(0, 0); }
  else if (a === "tagq") { e.stopPropagation(); ui.q = el.dataset.v; ui.stack = []; ui.view = "dump"; render(); }
  else if (a === "del") {
    if (!confirm("Delete this thought?")) return;
    const id = el.dataset.id;
    bank.thoughts = bank.thoughts.filter(t => t.id !== id);
    bank.deleted = { ...(bank.deleted || {}), [id]: new Date().toISOString() };
    ui.stack.pop(); render(); toast("Deleted"); changed();
  }
  else if (a === "edit") { ui.stack[ui.stack.length - 1].editing = true; render(); const ed = $("#edit"); ed.focus(); ed.setSelectionRange(ed.value.length, ed.value.length); }
  else if (a === "cancelEdit") { ui.stack[ui.stack.length - 1].editing = false; render(); }
  else if (a === "saveEdit") {
    const ed = $("#edit"), t = byId(ed.dataset.id), v = ed.value.trim();
    if (t && v && v !== t.text) { t.text = v; t.u = new Date().toISOString(); changed(); toast("Saved"); }
    ui.stack[ui.stack.length - 1].editing = false; render();
  }
  else if (a === "retryModel") { ui.model = "off"; loadModel(); render(); }
  else if (a === "export") exportMd();
  else if (a === "saveTok") {
    cfg.token = $("#tok").value.trim(); cfg.dataRepo = $("#repo").value.trim() || "idea-bank";
    writeLS(CFG_KEY, cfg); await sync(); render(); toast(ui.sync === "synced" ? "Synced" : ui.sync || "Saved");
  }
});
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") sync(); });

// ------------------------------------------------------------------ start
render(); status();
(async () => {
  try {
    const stored = await idb.all();
    for (const [id, { h, v }] of stored) { vecs.set(id, v instanceof Float32Array ? v : Float32Array.from(v)); vecHash.set(id, h); }
  } catch (e) { console.warn("idb", e); }
  sync();
  loadModel();
})();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
