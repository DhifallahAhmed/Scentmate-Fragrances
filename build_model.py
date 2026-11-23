"""build_model.py - fuse the trained pieces + the whole fragrance catalogue into ONE ONNX model.

Run after train.py (needs artifacts/). Writes site/model/ — the browser downloads and runs it with
ONNX Runtime Web, so no server is needed:
  config.json                    traits, accords, note vocabulary, ranking defaults, and the file names below
  files/recommender.<hash>.onnx  the full recommender (encoder + catalogue + filtering + ranking + top-K)
  files/items.<hash>.json        display data per fragrance id
The <hash> changes whenever the content changes, so browsers may cache those files forever
(see site/_headers); only the small config.json is re-checked on each visit.

MODEL INPUTS
  traits       float [1, T]  personality weights, order = config.traits (any scale; normalised inside)
  allowed      float [1, 3]  genders that may be returned: [women, men, unisex] as 0/1
  unique       float [1]     0 = popular first (>= 100 ratings), 1 = hidden gems (< 50 ratings)
  accord_pref  float [1, A]  soft accord preferences (+ liked / - disliked), order = config.accords
  must, avoid  float [1, A]  hard constraints (1 = result must / must not contain the accord)
  params       float [5]     [pop_weight, relevance_window, accord_weight, lowinfo_weight, anchor_weight]
  anchor       int64 [1]     fragrance id for "similar to ..." (used with anchor_weight > 0)
MODEL OUTPUTS
  ids [K] int64, scores [K], matches [K], trait_affinity [K, T], n_constrained [1], n_ranked [1]
"""
import hashlib, json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ART = Path("artifacts")
OUT = Path("site/model")
FILES = OUT / "files"

POPULAR_MIN, UNIQUE_MAX = 100, 50
DOMINANCE = [1.0, 0.85, 0.7, 0.55, 0.4]          # weight of an accord by its position in the list
GENDERS = ["women", "men", "unisex"]
TOP_K = 100                                     # candidates returned; the page re-ranks/trims
# Ranking defaults (written to config.json; tuned with evaluate.py --tune). Order = the `params` input.
PARAMS = {"pop_weight": 0.5, "relevance_window": 0.25, "accord_weight": 0.08, "lowinfo_weight": 0.0, "anchor_weight": 0.0}
if Path("eval/params.json").exists():                           # tuned values from `evaluate.py --tune`
    PARAMS.update(json.loads(Path("eval/params.json").read_text(encoding="utf-8")))


def load_mlp():
    """Pull the trained trait-encoder weights out of trait_encoder.onnx."""
    m = onnx.load(ART / "trait_encoder.onnx")
    w = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    return w["W1"], w["b1"], w["W2"], w["b2"]


def catalogue(items, accords):
    """Compact catalogue arrays + their exact float versions (what the graph reconstructs), for verify()."""
    N, A, a_idx = len(items), len(accords), {a: i for i, a in enumerate(accords)}
    E = np.load(ART / "fragrance_emb.npy").astype(np.float32)
    # 8-bit fingerprints, one scale per fragrance (top-20 overlap vs float32 ~97%, measured).
    scale = np.maximum(np.abs(E).max(axis=1), 1e-8) / 127.0
    E8 = np.clip(np.round(E / scale[:, None]), -127, 127).astype(np.int8)
    # Accord dominance as a level 0..5 (0 = absent) instead of a float -> 1 byte per cell.
    DTABLE = np.array([0.0] + DOMINANCE, np.float32)
    DLEV = np.zeros((N, A), np.uint8)
    for i, it in enumerate(items):
        for pos, a in enumerate(it["accords"]):
            if a in a_idx:
                DLEV[i, a_idx[a]] = min(pos, len(DOMINANCE) - 1) + 1
    return {
        "E8": E8, "E_scale": scale.astype(np.float32), "E_zp": np.zeros(N, np.int8),
        "DLEV": DLEV, "DTABLE": DTABLE,
        "GCODE": np.array([GENDERS.index(it["gender"]) for it in items], np.uint8),
        "COUNT": np.array([it["ratingCount"] for it in items], np.int32),
        "LOWINFO": np.array([it.get("lowInfo", False) for it in items], np.uint8),
    }


def build_graph(traits, accords, cat, N):
    W1, b1, W2, b2 = load_mlp()
    T, A = len(traits), len(accords)
    h = np.maximum(np.eye(T, dtype=np.float32) @ W1 + b1, 0)       # unit trait vectors (for explanations)
    QT = h @ W2 + b2
    QT /= np.linalg.norm(QT, axis=1, keepdims=True)
    count = cat["COUNT"].astype(np.float64)
    lo, hi = np.log1p(POPULAR_MIN), np.log1p(count.max())

    f = lambda v: np.array(v, np.float32)
    i64 = lambda v: np.array(v, np.int64)
    consts = dict(cat, W1=W1, b1=b1, W2=W2, b2=b2, QT=QT.astype(np.float32),
                  eps=f(1e-9), zero=f(0.0), one=f(1.0), half=f(0.5), big=f(1e4), hundred=f(100.0),
                  logmin=f(lo), logspan=f(hi - lo), popmin=f(POPULAR_MIN), unimax=f(UNIQUE_MAX),
                  gcodes=np.array([[0, 1, 2]], np.uint8), ax0=i64([0]), ax1=i64([1]), K=i64([min(TOP_K, N)]),
                  **{f"p{i}": np.array(i, np.int64) for i in range(5)})
    inits = [numpy_helper.from_array(v, k) for k, v in consts.items()]
    n = helper.make_node
    nodes = [
        # --- Unpack the compact catalogue
        n("DequantizeLinear", ["E8", "E_scale", "E_zp"], ["E"], axis=0),          # int8 -> float fingerprints
        n("Cast", ["DLEV"], ["dlev64"], to=TensorProto.INT64), n("Gather", ["DTABLE", "dlev64"], ["D"], axis=0),
        n("Greater", ["D", "zero"], ["P_b"]), n("Cast", ["P_b"], ["P"], to=TensorProto.FLOAT),
        n("Unsqueeze", ["GCODE", "ax1"], ["g_col"]), n("Equal", ["g_col", "gcodes"], ["G_b"]),
        n("Cast", ["G_b"], ["G"], to=TensorProto.FLOAT),
        n("Cast", ["COUNT"], ["cnt"], to=TensorProto.FLOAT), n("Add", ["cnt", "one"], ["cnt1"]), n("Log", ["cnt1"], ["logc"]),
        n("Sub", ["logc", "logmin"], ["lc0"]), n("Div", ["lc0", "logspan"], ["lc1"]),
        n("Max", ["lc1", "zero"], ["lc2"]), n("Min", ["lc2", "one"], ["POP"]),                  # 0..1 log popularity
        n("GreaterOrEqual", ["cnt", "popmin"], ["bp_b"]), n("Cast", ["bp_b"], ["BAND_POP"], to=TensorProto.FLOAT),
        n("Less", ["cnt", "unimax"], ["bu_b"]), n("Cast", ["bu_b"], ["BAND_UNI"], to=TensorProto.FLOAT),
        n("Sub", ["BAND_UNI", "BAND_POP"], ["BAND_DIFF"]),
        n("Cast", ["LOWINFO"], ["LOW"], to=TensorProto.FLOAT),
        # --- Ranking parameters
        n("Gather", ["params", "p0"], ["w_pop"]), n("Gather", ["params", "p1"], ["window"]),
        n("Gather", ["params", "p2"], ["w_acc"]), n("Gather", ["params", "p3"], ["w_low"]),
        n("Gather", ["params", "p4"], ["w_anchor"]),
        # --- 1. Personality -> query vector (trained network); traits normalised to sum 1 first
        n("ReduceSum", ["traits", "ax1"], ["tsum"], keepdims=1), n("Add", ["tsum", "eps"], ["tsum_e"]),
        n("Div", ["traits", "tsum_e"], ["traits_n"]),
        n("Gemm", ["traits_n", "W1", "b1"], ["h"]), n("Relu", ["h"], ["hr"]), n("Gemm", ["hr", "W2", "b2"], ["q"]),
        n("Mul", ["q", "q"], ["q2"]), n("ReduceSum", ["q2", "ax1"], ["ss"], keepdims=1),
        n("Sqrt", ["ss"], ["nrm"]), n("Add", ["nrm", "eps"], ["nrm_e"]), n("Div", ["q", "nrm_e"], ["qn"]),
        # --- 1b. "Similar to <anchor>": blend in that fragrance's own fingerprint, renormalise
        n("Gather", ["E", "anchor"], ["e_anchor"], axis=0),
        n("Sub", ["one", "w_anchor"], ["w_self"]), n("Mul", ["qn", "w_self"], ["qa"]), n("Mul", ["e_anchor", "w_anchor"], ["qb"]),
        n("Add", ["qa", "qb"], ["qm"]), n("Mul", ["qm", "qm"], ["qm2"]), n("ReduceSum", ["qm2", "ax1"], ["qss"], keepdims=1),
        n("Sqrt", ["qss"], ["qnrm"]), n("Add", ["qnrm", "eps"], ["qnrm_e"]), n("Div", ["qm", "qnrm_e"], ["qf"]),
        # --- 2. Personal match = cosine with every fragrance + soft accord preferences
        n("Gemm", ["E", "qf"], ["cos2"], transB=1),
        n("Gemm", ["D", "accord_pref"], ["acc2"], transB=1), n("Mul", ["acc2", "w_acc"], ["accw"]),
        n("Add", ["cos2", "accw"], ["match2"]), n("Squeeze", ["match2", "ax1"], ["match"]),
        # --- 3. Filters: gender, popularity band, hard accord constraints
        n("Gemm", ["G", "allowed"], ["g2"], transB=1), n("Squeeze", ["g2", "ax1"], ["gok"]),
        n("Mul", ["BAND_DIFF", "unique"], ["bd"]), n("Add", ["BAND_POP", "bd"], ["band"]),
        n("Sub", ["avoid", "must"], ["am"]), n("Gemm", ["P", "am"], ["v2"], transB=1),
        n("Squeeze", ["v2", "ax1"], ["vp"]), n("ReduceSum", ["must", "ax1"], ["msum"], keepdims=0),
        n("Add", ["vp", "msum"], ["viol"]), n("Less", ["viol", "half"], ["cons_b"]),
        n("Cast", ["cons_b"], ["cons"], to=TensorProto.FLOAT),
        n("Mul", ["gok", "band"], ["gb"]), n("Mul", ["gb", "cons"], ["valid"]),
        # --- 4. Relevance gate (popular mode): only items within `window` of the best valid match
        n("Sub", ["valid", "one"], ["vm1"]), n("Mul", ["vm1", "big"], ["vpen"]), n("Add", ["match", "vpen"], ["masked"]),
        n("ReduceMax", ["masked"], ["best"], axes=[0], keepdims=1), n("Sub", ["best", "window"], ["thr"]),
        n("GreaterOrEqual", ["match", "thr"], ["gate_b"]), n("Cast", ["gate_b"], ["gate_f"], to=TensorProto.FLOAT),
        n("Max", ["gate_f", "unique"], ["gate"]),
        # --- 5. Score = match + popularity (popular mode) - low-information penalty (fades as popularity grows)
        n("Sub", ["one", "unique"], ["not_u"]), n("Mul", ["POP", "w_pop"], ["popw"]), n("Mul", ["popw", "not_u"], ["popterm"]),
        n("Sub", ["one", "POP"], ["unpop"]), n("Mul", ["LOW", "unpop"], ["low1"]), n("Mul", ["low1", "w_low"], ["lowpen"]),
        n("Add", ["match", "popterm"], ["f0"]), n("Sub", ["f0", "lowpen"], ["final"]),
        n("Mul", ["valid", "gate"], ["keep"]), n("Mul", ["vm1", "big"], ["pen_invalid"]),
        n("Sub", ["gate", "one"], ["gm1"]), n("Mul", ["gm1", "valid"], ["gm1v"]), n("Mul", ["gm1v", "hundred"], ["pen_gate"]),
        n("Add", ["final", "pen_invalid"], ["f1"]), n("Add", ["f1", "pen_gate"], ["score"]),
        # --- 6. Top-K + details
        n("Unsqueeze", ["score", "ax0"], ["score2"]),
        n("TopK", ["score2", "K"], ["vals2", "ids2"], axis=-1, largest=1, sorted=1),
        n("Squeeze", ["ids2", "ax0"], ["ids"]), n("Squeeze", ["vals2", "ax0"], ["scores"]),
        n("Gather", ["match", "ids"], ["matches"], axis=0),
        n("Gather", ["E", "ids"], ["Etop"], axis=0), n("Gemm", ["Etop", "QT"], ["trait_affinity"], transB=1),
        n("ReduceSum", ["valid", "ax0"], ["n_constrained"], keepdims=1),
        n("ReduceSum", ["keep", "ax0"], ["n_ranked"], keepdims=1),
    ]
    fin = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(
        nodes, "scentmate_recommender",
        [fin("traits", [1, T]), fin("allowed", [1, 3]), fin("unique", [1]), fin("accord_pref", [1, A]),
         fin("must", [1, A]), fin("avoid", [1, A]), fin("params", [5]),
         helper.make_tensor_value_info("anchor", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [None]), fin("scores", [None]), fin("matches", [None]),
         fin("trait_affinity", [None, T]), fin("n_constrained", [1]), fin("n_ranked", [1])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="scentmate")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def write_hashed(name, ext, data: bytes):
    h = hashlib.sha256(data).hexdigest()[:10]
    (FILES / f"{name}.{h}.{ext}").write_bytes(data)
    return f"files/{name}.{h}.{ext}"


def build():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    traits, items = meta["traits"], meta["items"]
    N = len(items)
    accords = sorted({a for it in items for a in it["accords"] if a != "Main Accords"})
    cat = catalogue(items, accords)
    model = build_graph(traits, accords, cat, N)

    FILES.mkdir(parents=True, exist_ok=True)
    for old in list(FILES.glob("*")) + [OUT / "recommender.onnx", OUT / "items.json"]:   # previous builds
        old.unlink(missing_ok=True)
    model_file = write_hashed("recommender", "onnx", model.SerializeToString())

    from collections import Counter
    from train import key_hash
    hashes = [key_hash(it["key"]) for it in items]
    assert len(set(hashes)) == len(hashes), "fragrance key hash collision — switch to a 64-bit hash"
    # [name, brand, accords, notes, gender, ratings, estimated?, keyHash]
    items_file = write_hashed("items", "json", json.dumps(
        [[it["name"], it["brand"], it["accords"][:5], it["notes"][:5], it["gender"][0], it["ratingCount"],
          int(it.get("ratingSource") == "estimated"), hashes[i]] for i, it in enumerate(items)],
        ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    note_counts = Counter(nt.lower() for it in items for nt in it["notes"])
    (OUT / "config.json").write_text(json.dumps(
        {"traits": traits, "accords": accords, "genders": GENDERS, "top_k": int(min(TOP_K, N)),
         "params": PARAMS, "param_order": list(PARAMS),
         "files": {"model": model_file, "items": items_file},
         "notes": sorted(n for n, c in note_counts.items() if c >= 20 and len(n) >= 4)},
        ensure_ascii=False), encoding="utf-8")
    size = (OUT / model_file).stat().st_size / 1e6
    print(f"{model_file}: {N} fragrances, {len(accords)} accords, {size:.1f} MB")
    return traits, accords, items, cat


def reference(cat, traits_w, allowed, unique, pref, must, avoid, params, anchor, W):
    """Plain-numpy version of the graph, used to verify the ONNX model."""
    W1, b1, W2, b2 = W
    E = cat["E8"].astype(np.float32) * cat["E_scale"][:, None]
    D = np.array([0.0] + DOMINANCE, np.float32)[cat["DLEV"]]
    P = (D > 0).astype(np.float32)
    count = cat["COUNT"].astype(np.float64)
    lo, hi = np.log1p(POPULAR_MIN), np.log1p(count.max())
    POP = np.clip((np.log1p(count) - lo) / (hi - lo), 0, 1)
    w_pop, window, w_acc, w_low, w_anchor = params
    t = traits_w / (traits_w.sum() + 1e-9)
    q = np.maximum(t @ W1 + b1, 0) @ W2 + b2; q /= np.linalg.norm(q) + 1e-9
    q = q * (1 - w_anchor) + E[anchor] * w_anchor; q /= np.linalg.norm(q) + 1e-9
    match = E @ q + w_acc * (D @ pref)
    ok = allowed[cat["GCODE"]] > 0
    ok &= (count < UNIQUE_MAX) if unique else (count >= POPULAR_MIN)
    ok &= (P @ must == must.sum()) & (P @ avoid == 0)
    if not unique:
        ok &= match >= match[ok].max() - window
    final = match + (0 if unique else w_pop * POP) - w_low * cat["LOWINFO"] * (1 - POP)
    return np.where(ok, final, -np.inf)


def verify(traits, accords, items, cat):
    """Check the ONNX model against the numpy reference (random personas, filters, anchors)."""
    sess = ort.InferenceSession(str(OUT / json.loads((OUT / "config.json").read_text())["files"]["model"]),
                                providers=["CPUExecutionProvider"])
    W = load_mlp()
    rng = np.random.default_rng(1)
    ok_all = True
    for case in range(8):
        t = (rng.dirichlet(np.ones(len(traits)) * .5) * (3.7 if case % 2 else 1)).astype(np.float32)
        allowed = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]][case % 4], np.float32)
        unique = float(case == 5)
        pref, must, avoid = (np.zeros(len(accords), np.float32) for _ in range(3))
        if case >= 2: pref[accords.index("Spicy")] = 1.5; must[accords.index("Spicy")] = 1
        if case >= 3: avoid[accords.index("Sweet")] = 1
        params = np.array([0.5, 0.25, 0.08, 0.05 * (case % 3), 0.7 if case >= 6 else 0.0], np.float32)
        anchor = int(rng.integers(len(items)))
        out = sess.run(None, {"traits": t[None], "allowed": allowed[None], "unique": np.array([unique], np.float32),
                              "accord_pref": pref[None], "must": must[None], "avoid": avoid[None],
                              "params": params, "anchor": np.array([anchor], np.int64)})
        scored = reference(cat, t, allowed, unique, pref, must, avoid, params, anchor, W)
        ref, got = scored.argsort()[::-1][:5], out[0][:5]
        same = list(got) == list(ref) or np.abs(scored[got] - scored[ref]).max() < 1e-5
        ok_all &= bool(same)
        print(f"case {case}: onnx {got.tolist()}  numpy {ref.tolist()}  "
              f"max score diff {np.abs(out[1][:5] - scored[got]).max():.1e}  {'OK' if same else 'MISMATCH'}")
    assert ok_all, "ONNX model does not match the numpy reference"


if __name__ == "__main__":
    verify(*build())
    # Quality gate: score the freshly built model on the hand-written personas (eval/personas.json).
    # Fails (exit 1) on a broken rule or a drop vs eval/baseline.json. Accept an intended change with:
    #   python evaluate.py --update-baseline
    import evaluate
    raise SystemExit(evaluate.main())
