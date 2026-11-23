/* worker.js — the AI model, running in a background thread so the page never stutters.
   Loads ONNX Runtime Web + the model + the catalogue once, then answers messages:
     {id, type: "recommend", payload: req}  -> ranked fragrances (same logic as evaluate.py)
     {id, type: "search",    payload: text} -> fragrances whose name/brand match the text
   On start it posts {type: "ready", ...} with the vocabulary the page needs, or {type: "failed"}. */
importScripts("vendor/ort/1.31.1/ort.wasm.min.js");

const GENDER_CODE = { w: "women", m: "men", u: "unisex" };
const ALLOWED = { women: [1, 0, 1], men: [0, 1, 1], unisex: [0, 0, 1] };   // [women, men, unisex] may be returned
let CFG, ITEMS, NOTES_LOWER, SEARCH_TEXT, SESSION, TRAITS, ACCORDS;

async function okJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return r.json();
}

const fold = s => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();   // "Mâle" -> "male"

async function load() {
  const t0 = performance.now();
  CFG = await okJson("model/config.json");             // small; always re-checked (see _headers)
  ort.env.wasm.wasmPaths = new URL("vendor/ort/1.31.1/", self.location.href).href;
  ort.env.wasm.numThreads = 1;                         // threads need cross-origin isolation; 1 is plenty here
  [ITEMS, SESSION] = await Promise.all([
    okJson("model/" + CFG.files.items),                // content-hashed names: cached forever by browsers
    ort.InferenceSession.create("model/" + CFG.files.model, { executionProviders: ["wasm"] }),
  ]);
  TRAITS = CFG.traits; ACCORDS = CFG.accords;
  NOTES_LOWER = ITEMS.map(it => it[3].map(n => n.toLowerCase()));
  SEARCH_TEXT = ITEMS.map(it => " " + fold(it[0] + " " + it[1]).replace(/[^a-z0-9]+/g, " ") + " ");
  postMessage({ type: "ready", traits: TRAITS, accords: ACCORDS, notes: CFG.notes, loadMs: Math.round(performance.now() - t0) });
}

/* One pass through the ONNX model. */
async function runModel(req, must, avoid) {
  const T = TRAITS.length, A = ACCORDS.length;
  const t = Float32Array.from(TRAITS, k => Math.max(0, +req.traits[k] || 0));
  if (!t.some(v => v > 0)) t.fill(1);                  // no personality info yet -> neutral blend
  const vec = f => Float32Array.from(ACCORDS, f);
  const p = CFG.params;
  const out = await SESSION.run({
    traits:      new ort.Tensor("float32", t, [1, T]),
    allowed:     new ort.Tensor("float32", Float32Array.from(ALLOWED[req.gender] || [1, 1, 1]), [1, 3]),
    unique:      new ort.Tensor("float32", Float32Array.of(req.unique ? 1 : 0), [1]),
    accord_pref: new ort.Tensor("float32", vec(a => +req.accord_prefs[a] || 0), [1, A]),
    must:        new ort.Tensor("float32", vec(a => (must.includes(a) ? 1 : 0)), [1, A]),
    avoid:       new ort.Tensor("float32", vec(a => (avoid.includes(a) ? 1 : 0)), [1, A]),
    params:      new ort.Tensor("float32", Float32Array.of(p.pop_weight, p.relevance_window, p.accord_weight,
                                                            p.lowinfo_weight, req.anchor != null ? (req.anchor_weight ?? 0.85) : 0), [5]),
    anchor:      new ort.Tensor("int64", BigInt64Array.of(BigInt(req.anchor ?? 0)), [1]),
  });
  return {
    ids: Array.from(out.ids.data, Number), scores: out.scores.data, matches: out.matches.data,
    affinity: out.trait_affinity.data, nCons: out.n_constrained.data[0], traitVec: t,
  };
}

/* Relax impossible wishes, re-rank by notes, skip what was already shown. */
async function recommend(req) {
  const t0 = performance.now();
  let cons = req.constraints.filter(c => ACCORDS.includes(c.accord));
  let out;
  while (true) {                                       // drop the OLDEST wish until something qualifies
    out = await runModel(req, cons.filter(c => c.type === "must").map(c => c.accord),
                              cons.filter(c => c.type === "avoid").map(c => c.accord));
    if (out.nCons >= req.top_k || cons.length === 0) break;
    cons = cons.slice(1);
  }
  const relaxed = req.constraints.filter(c => !cons.some(k => k.type === c.type && k.accord === c.accord));

  const T = TRAITS.length;
  const notes = Object.entries(req.notes).map(([k, v]) => [k.toLowerCase(), v]);
  // "Similar to X": hide X itself and its other versions (same brand + same first word), show alternatives.
  const anchorLine = req.anchor != null ? lineOf(req.anchor) : null;
  const n = Math.max(1, Math.min(out.nCons, out.ids.length));
  const cands = [];
  for (let j = 0; j < n; j++) {
    const i = out.ids[j];
    if (anchorLine && lineOf(i) === anchorLine) continue;
    const nb = notes.reduce((s, [note, v]) => s + (NOTES_LOWER[i].includes(note) ? 0.1 * v : 0), 0);
    cands.push({ i, score: out.scores[j] + nb, match: out.matches[j] + nb, aff: out.affinity.subarray(j * T, (j + 1) * T) });
  }
  cands.sort((a, b) => b.score - a.score);
  const excl = new Set(req.exclude);
  const fresh = cands.filter(c => !excl.has(c.i));
  const reset = fresh.length < req.top_k && cands.length >= req.top_k;
  const picked = (reset ? cands : fresh).slice(0, req.top_k);
  return {
    relaxed, reset_shown: reset, ms: Math.round(performance.now() - t0),
    results: picked.map(c => {
      const [name, brand, accords, noteList, g, count, estimated, keyHash] = ITEMS[c.i];
      let best = 0;                                    // strongest weighted trait -> "fits your romantic side"
      for (let k = 1; k < T; k++) if (out.traitVec[k] * c.aff[k] > out.traitVec[best] * c.aff[best]) best = k;
      return { id: c.i, keyHash, name, brand, accords, notes: noteList, gender: GENDER_CODE[g], ratingCount: count,
               estimated: !!estimated, match: c.match, score: c.score, topTrait: TRAITS[best] };
    }),
  };
}

function lineOf(i) {
  const [name, brand] = ITEMS[i];
  return fold(brand) + "|" + fold(name).split(/[^a-z0-9]+/).filter(Boolean)[0];
}

/* "dior sauvage" -> best matching fragrances. Every word must appear in name+brand; ties go to the
   most-rated (people mean "Sauvage EDT", not a limited edition). */
function search(text) {
  const words = fold(text).replace(/[^a-z0-9]+/g, " ").trim().split(" ").filter(w => w.length > 1);
  if (!words.length) return [];
  const hits = [];
  for (let i = 0; i < ITEMS.length; i++) {
    const s = SEARCH_TEXT[i];
    if (words.every(w => s.includes(" " + w))) hits.push(i);
  }
  hits.sort((a, b) => ITEMS[b][5] - ITEMS[a][5] || ITEMS[a][0].length - ITEMS[b][0].length);
  return hits.slice(0, 5).map(i => ({ id: i, name: ITEMS[i][0], brand: ITEMS[i][1], gender: GENDER_CODE[ITEMS[i][4]],
                                      accords: ITEMS[i][2], ratingCount: ITEMS[i][5] }));
}

const ready = load().catch(err => { postMessage({ type: "failed", error: String(err && err.message || err) }); throw err; });

onmessage = async e => {
  const { id, type, payload } = e.data;
  try {
    await ready;
    const result = type === "recommend" ? await recommend(payload) : type === "search" ? search(payload) : null;
    postMessage({ id, ok: true, result });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
