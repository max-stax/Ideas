// Pure logic shared by the app and the tests: merging, similarity, themes.
// No DOM, no network.

// ---------------------------------------------------------------- merge
// A bank is { thoughts: [{id, text, created, u, source}], deleted: {id: time} }.
// Two copies (phone + repo) merge by id: newest edit wins, deletions win over older edits.
export function merge(a, b) {
  const deleted = { ...(a?.deleted || {}) };
  for (const [id, t] of Object.entries(b?.deleted || {})) if (!deleted[id] || t > deleted[id]) deleted[id] = t;
  const byId = new Map();
  for (const t of [...(a?.thoughts || []), ...(b?.thoughts || [])]) {
    if (!t || !t.id) continue;
    const cur = byId.get(t.id);
    if (!cur || stamp(t) > stamp(cur)) byId.set(t.id, t);
  }
  const thoughts = [...byId.values()]
    .filter(t => !(deleted[t.id] && deleted[t.id] >= stamp(t)))
    .sort((x, y) => (y.created || "").localeCompare(x.created || ""));
  return { thoughts, deleted };
}
const stamp = t => t.u || t.created || "";

// Same content, ignoring order — decides whether a push is needed.
export function sameBank(a, b) {
  const key = x => JSON.stringify([
    (x?.thoughts || []).map(t => [t.id, stamp(t), t.text]).sort((p, q) => p[0].localeCompare(q[0])),
    Object.entries(x?.deleted || {}).sort(),
  ]);
  return key(a) === key(b);
}

export function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

// ---------------------------------------------------------------- text
export const tagsOf = text => [...new Set((text.match(/#[\p{L}\p{N}_-]+/gu) || []).map(t => t.toLowerCase()))];

export function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(36);
}

const STOP = new Set(`a about above after again against all am an and any are aren't as at be because been before being below between
both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further get gets got
had hadn't has hasn't have haven't having he her here hers herself him himself his how i i'm i've if in into is isn't it it's its itself
just let's like me more most much my myself no nor not now of off on once only or other ought our ours ourselves out over own really
same she should shouldn't so some such than that that's the their theirs them themselves then there there's these they they're this
those through to too under until up us very was wasn't we we're were weren't what what's when where which while who whom why will with
won't would wouldn't you you're your yours yourself yourselves thing things something stuff one ones make makes making way ways also
even still yet ever every lot lots maybe kind sort actually basically going gonna want wants need needs think thinks know people
person feel feels time times day days good bad`.split(/\s+/));

export function words(text) {
  return (text.toLowerCase().replace(/#[\p{L}\p{N}_-]+/gu, " ").match(/[\p{L}][\p{L}'-]{2,}/gu) || [])
    .map(w => w.replace(/'s$/, "")).filter(w => !STOP.has(w) && w.length > 2);
}

// ---------------------------------------------------------------- vectors
// Vectors come from the model already normalized, so cosine = dot product.
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const RELATED_MIN = 0.35;

// Nearest thoughts to `id`, best first.
export function related(id, items, vecs, k = 5, min = RELATED_MIN) {
  const v = vecs.get(id);
  if (!v) return [];
  const out = [];
  for (const t of items) {
    if (t.id === id) continue;
    const w = vecs.get(t.id);
    if (!w) continue;
    const s = dot(v, w);
    if (s >= min) out.push({ t, s });
  }
  return out.sort((x, y) => y.s - x.s).slice(0, k);
}

// ---------------------------------------------------------------- themes
// Average-linkage clustering: keep merging the two groups whose thoughts are,
// on average, most alike, until no pair is alike enough. Average linkage avoids
// the "chain" effect where A~B~C~D ends up one blob even though A and D are unrelated.
export const THEME_MIN = 0.42;

export function clusterIds(ids, vecs, threshold = THEME_MIN) {
  const n = ids.length;
  if (n < 2) return ids.map(id => [id]);
  const V = ids.map(id => vecs.get(id));
  const S = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const s = dot(V[i], V[j]); S[i][j] = s; S[j][i] = s; }
  const size = new Array(n).fill(1);
  const alive = new Array(n).fill(true);
  const members = ids.map(id => [id]);
  // best partner per row, so each step is O(n) instead of O(n²)
  const best = new Int32Array(n).fill(-1);
  const rowBest = i => {
    let b = -1, bs = -Infinity;
    for (let j = 0; j < n; j++) if (j !== i && alive[j] && S[i][j] > bs) { bs = S[i][j]; b = j; }
    best[i] = b;
  };
  for (let i = 0; i < n; i++) rowBest(i);
  for (;;) {
    let a = -1, as = -Infinity;
    for (let i = 0; i < n; i++) if (alive[i] && best[i] >= 0 && S[i][best[i]] > as) { as = S[i][best[i]]; a = i; }
    if (a < 0 || as < threshold) break;
    const b = best[a];
    // fold b into a (Lance–Williams update for average linkage)
    for (let k = 0; k < n; k++) {
      if (!alive[k] || k === a || k === b) continue;
      const s = (S[a][k] * size[a] + S[b][k] * size[b]) / (size[a] + size[b]);
      S[a][k] = s; S[k][a] = s;
    }
    size[a] += size[b]; alive[b] = false; members[a].push(...members[b]); members[b] = [];
    for (let i = 0; i < n; i++) if (alive[i] && (i === a || best[i] === a || best[i] === b)) rowBest(i);
    for (let i = 0; i < n; i++) if (alive[i] && i !== a && S[i][a] > (best[i] >= 0 ? S[i][best[i]] : -Infinity)) best[i] = a;
  }
  return members.filter(m => m.length);
}

// Label each group with the words that are common in it but not everywhere.
export function labelThemes(groups, textById, maxWords = 3) {
  const docFreq = new Map();
  const groupWords = groups.map(g => {
    const counts = new Map();
    for (const id of g) for (const w of new Set(words(textById.get(id) || ""))) counts.set(w, (counts.get(w) || 0) + 1);
    for (const w of counts.keys()) docFreq.set(w, (docFreq.get(w) || 0) + 1);
    return counts;
  });
  const G = groups.length;
  return groupWords.map((counts, gi) => {
    const size = groups[gi].length;
    const scored = [...counts.entries()]
      .filter(([, c]) => size < 3 || c >= 2)
      .map(([w, c]) => [w, (c / size) * Math.log(1 + G / docFreq.get(w))])
      .sort((x, y) => y[1] - x[1]);
    return scored.slice(0, maxWords).map(([w]) => w);
  });
}

// The thought that best represents a group (highest average similarity to the rest).
export function centerOf(group, vecs) {
  if (group.length <= 2) return group[0];
  let best = group[0], bs = -Infinity;
  for (const a of group) {
    let s = 0;
    for (const b of group) if (a !== b) s += dot(vecs.get(a), vecs.get(b));
    if (s > bs) { bs = s; best = a; }
  }
  return best;
}

export function themes(items, vecs) {
  const withVec = items.filter(t => vecs.has(t.id));
  const groups = clusterIds(withVec.map(t => t.id), vecs);
  const multi = groups.filter(g => g.length >= 2);
  const textById = new Map(items.map(t => [t.id, t.text]));
  const labels = labelThemes(multi, textById);
  const out = multi.map((g, i) => {
    const center = centerOf(g, vecs);
    return { ids: [center, ...g.filter(id => id !== center)], label: labels[i] };
  }).sort((x, y) => y.ids.length - x.ids.length);
  return { themes: out, loose: groups.filter(g => g.length === 1).map(g => g[0]) };
}
