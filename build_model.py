"""build_model.py - fuse the trained pieces + the whole fragrance catalogue into ONE ONNX model.

Run after train.py (needs artifacts/). Writes site/model/ — the browser downloads and runs it
(ONNX Runtime Web), so no server is needed:
  recommender.onnx  the full recommender (encoder + catalogue + filtering + ranking + top-K)
  items.json        display data for each fragrance id (name, brand, accords, notes, gender, ratings)
  config.json       trait / accord order expected by the model inputs, plus the note vocabulary

MODEL INPUTS (all float32)
  traits        [1, 8]  personality weights, order = config.traits
  allowed       [1, 3]  which genders may be returned: [women, men, unisex] as 0/1
  unique        [1]     0 = popular first (>= 100 ratings), 1 = hidden gems (< 50 ratings)
  accord_pref   [1, A]  soft accord preferences (+ liked / - disliked), order = config.accords
  must          [1, A]  1 = result MUST contain this accord
  avoid         [1, A]  1 = result must NOT contain this accord
MODEL OUTPUTS
  ids [K] int64, scores [K], matches [K], trait_affinity [K, 8], n_constrained [1], n_ranked [1]
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ART = Path("artifacts")
OUT = Path("site/model")

# Ranking rules
POPULAR_MIN, UNIQUE_MAX = 100, 50
POP_WEIGHT, RELEVANCE_WINDOW, ACCORD_WEIGHT = 0.5, 0.25, 0.08
DOMINANCE = [1.0, 0.85, 0.7, 0.55, 0.4]          # weight of an accord by its position in the list
GENDERS = ["women", "men", "unisex"]
TOP_K = 100                                     # candidates returned; the page re-ranks/trims


def load_mlp():
    """Pull the trained trait-encoder weights out of trait_encoder.onnx."""
    m = onnx.load(ART / "trait_encoder.onnx")
    w = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    return w["W1"], w["b1"], w["W2"], w["b2"]


def build():
    E = np.load(ART / "fragrance_emb.npy").astype(np.float32)          # (N, 64) unit rows
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    traits, items = meta["traits"], meta["items"]
    N = len(items)
    W1, b1, W2, b2 = load_mlp()

    # ---- Catalogue matrices baked into the model ------------------------------------------------
    accords = sorted({a for it in items for a in it["accords"] if a != "Main Accords"})
    A, a_idx = len(accords), {a: i for i, a in enumerate(accords)}
    D = np.zeros((N, A), np.float32)       # accord dominance (for soft preferences)
    P = np.zeros((N, A), np.float32)       # accord presence (for hard must/avoid constraints)
    G = np.zeros((N, 3), np.float32)       # gender one-hot
    for i, it in enumerate(items):
        for pos, a in enumerate(it["accords"]):
            if a in a_idx:
                D[i, a_idx[a]] = DOMINANCE[min(pos, len(DOMINANCE) - 1)]
                P[i, a_idx[a]] = 1.0
        G[i, GENDERS.index(it["gender"])] = 1.0
    count = np.array([it["ratingCount"] for it in items], np.float64)
    lo, hi = np.log1p(POPULAR_MIN), np.log1p(count.max())
    POP = np.clip((np.log1p(count) - lo) / (hi - lo), 0, 1).astype(np.float32)
    BAND_POP = (count >= POPULAR_MIN).astype(np.float32)
    BAND_DIFF = (count < UNIQUE_MAX).astype(np.float32) - BAND_POP     # band = BAND_POP + unique * BAND_DIFF

    # Unit trait vectors in scent space (for the "fits your romantic side" explanations).
    h = np.maximum(np.eye(len(traits), dtype=np.float32) @ W1 + b1, 0)
    QT = h @ W2 + b2
    QT /= np.linalg.norm(QT, axis=1, keepdims=True)

    f = lambda v: np.array(v, np.float32)
    i64 = lambda v: np.array(v, np.int64)
    # Size for the web: the two big matrices are stored as float16 (cast back to float32 inside the
    # graph) and accord PRESENCE is derived from dominance (P = D > 0) instead of being stored.
    # 18.7 MB -> ~8 MB. Verification below uses the same rounded values, so results must match exactly.
    E16, D16 = E.astype(np.float16), D.astype(np.float16)
    consts = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "E16": E16, "D16": D16, "G": G, "QT": QT.astype(np.float32),
              "POP": POP, "BAND_POP": BAND_POP, "BAND_DIFF": BAND_DIFF,
              "eps": f(1e-9), "zero": f(0.0), "one": f(1.0), "half": f(0.5), "big": f(1e4), "hundred": f(100.0),
              "w_acc": f(ACCORD_WEIGHT), "w_pop": f(POP_WEIGHT), "window": f(RELEVANCE_WINDOW),
              "ax0": i64([0]), "ax1": i64([1]), "K": i64([min(TOP_K, N)])}
    inits = [numpy_helper.from_array(v, k) for k, v in consts.items()]

    n = helper.make_node
    nodes = [
        # Unpack the catalogue: float16 -> float32, presence = dominance > 0.
        n("Cast", ["E16"], ["E"], to=TensorProto.FLOAT), n("Cast", ["D16"], ["D"], to=TensorProto.FLOAT),
        n("Greater", ["D", "zero"], ["P_b"]), n("Cast", ["P_b"], ["P"], to=TensorProto.FLOAT),
        # 0. Scale trait weights to sum to 1 — the network was trained on mixes that sum to 1,
        #    so {"romantic": 1.6, "cozy": 0.6} and {"romantic": 0.73, "cozy": 0.27} mean the same thing.
        n("ReduceSum", ["traits", "ax1"], ["tsum"], keepdims=1), n("Add", ["tsum", "eps"], ["tsum_e"]),
        n("Div", ["traits", "tsum_e"], ["traits_n"]),
        # 1. Personality -> query vector (the trained neural network), then L2-normalise.
        n("Gemm", ["traits_n", "W1", "b1"], ["h"]), n("Relu", ["h"], ["hr"]),
        n("Gemm", ["hr", "W2", "b2"], ["q"]),
        n("Mul", ["q", "q"], ["q2"]), n("ReduceSum", ["q2", "ax1"], ["ss"], keepdims=1),
        n("Sqrt", ["ss"], ["nrm"]), n("Add", ["nrm", "eps"], ["nrm_e"]), n("Div", ["q", "nrm_e"], ["qn"]),
        # 2. Cosine similarity with every fragrance + soft accord preferences = personal match.
        n("Gemm", ["E", "qn"], ["cos2"], transB=1),
        n("Gemm", ["D", "accord_pref"], ["acc2"], transB=1), n("Mul", ["acc2", "w_acc"], ["accw"]),
        n("Add", ["cos2", "accw"], ["match2"]), n("Squeeze", ["match2", "ax1"], ["match"]),
        # 3. Filters: gender, popularity band, hard accord constraints.
        n("Gemm", ["G", "allowed"], ["g2"], transB=1), n("Squeeze", ["g2", "ax1"], ["gok"]),
        n("Mul", ["BAND_DIFF", "unique"], ["bd"]), n("Add", ["BAND_POP", "bd"], ["band"]),
        #    violations = (#required accords missing) + (#forbidden accords present)
        #               = sum(must) - P·must + P·avoid = sum(must) + P·(avoid - must)
        n("Sub", ["avoid", "must"], ["am"]), n("Gemm", ["P", "am"], ["v2"], transB=1),
        n("Squeeze", ["v2", "ax1"], ["vp"]), n("ReduceSum", ["must", "ax1"], ["msum"], keepdims=0),
        n("Add", ["vp", "msum"], ["viol"]), n("Less", ["viol", "half"], ["cons_b"]),
        n("Cast", ["cons_b"], ["cons"], to=TensorProto.FLOAT),
        n("Mul", ["gok", "band"], ["gb"]), n("Mul", ["gb", "cons"], ["valid"]),
        # 4. Relevance gate (popular mode): only items within RELEVANCE_WINDOW of the best valid match.
        n("Sub", ["valid", "one"], ["vm1"]), n("Mul", ["vm1", "big"], ["vpen"]), n("Add", ["match", "vpen"], ["masked"]),
        n("ReduceMax", ["masked"], ["best"], axes=[0], keepdims=1), n("Sub", ["best", "window"], ["thr"]),
        n("GreaterOrEqual", ["match", "thr"], ["gate_b"]), n("Cast", ["gate_b"], ["gate_f"], to=TensorProto.FLOAT),
        n("Max", ["gate_f", "unique"], ["gate"]),                       # unique mode: no gate
        # 5. Final score: match + popularity boost (popular mode only); invalid items pushed to -1e4.
        n("Sub", ["one", "unique"], ["not_u"]), n("Mul", ["POP", "w_pop"], ["popw"]), n("Mul", ["popw", "not_u"], ["popterm"]),
        n("Add", ["match", "popterm"], ["final"]),
        #    Penalties: invalid items -1e4 (never shown); valid items outside the relevance gate -100,
        #    so they rank after all gated items but can still be used for "show me others".
        n("Mul", ["valid", "gate"], ["keep"]), n("Mul", ["vm1", "big"], ["pen_invalid"]),
        n("Sub", ["gate", "one"], ["gm1"]), n("Mul", ["gm1", "valid"], ["gm1v"]), n("Mul", ["gm1v", "hundred"], ["pen_gate"]),
        n("Add", ["final", "pen_invalid"], ["f1"]), n("Add", ["f1", "pen_gate"], ["score"]),
        # 6. Top-K and per-result details.
        n("Unsqueeze", ["score", "ax0"], ["score2"]),
        n("TopK", ["score2", "K"], ["vals2", "ids2"], axis=-1, largest=1, sorted=1),
        n("Squeeze", ["ids2", "ax0"], ["ids"]), n("Squeeze", ["vals2", "ax0"], ["scores"]),
        n("Gather", ["match", "ids"], ["matches"], axis=0),
        n("Gather", ["E", "ids"], ["Etop"], axis=0), n("Gemm", ["Etop", "QT"], ["trait_affinity"], transB=1),
        n("ReduceSum", ["valid", "ax0"], ["n_constrained"], keepdims=1),
        n("ReduceSum", ["keep", "ax0"], ["n_ranked"], keepdims=1),
    ]
    T = len(traits)
    fin = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(
        nodes, "scentmate_recommender",
        [fin("traits", [1, T]), fin("allowed", [1, 3]), fin("unique", [1]),
         fin("accord_pref", [1, A]), fin("must", [1, A]), fin("avoid", [1, A])],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [None]), fin("scores", [None]), fin("matches", [None]),
         fin("trait_affinity", [None, T]), fin("n_constrained", [1]), fin("n_ranked", [1])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="scentmate")
    model.ir_version = 9
    onnx.checker.check_model(model)

    OUT.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT / "recommender.onnx")

    # Display data per id + note vocabulary (notes used by >= 20 fragrances) for the chat's parser.
    from collections import Counter
    note_counts = Counter(nt.lower() for it in items for nt in it["notes"])
    (OUT / "items.json").write_text(json.dumps(
        [[it["name"], it["brand"], it["accords"][:5], it["notes"][:5], it["gender"][0], it["ratingCount"]] for it in items],
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps(
        {"traits": traits, "accords": accords, "genders": GENDERS, "top_k": int(min(TOP_K, N)),
         "notes": sorted(n for n, c in note_counts.items() if c >= 20 and len(n) >= 4)},
        ensure_ascii=False), encoding="utf-8")
    size = (OUT / "recommender.onnx").stat().st_size / 1e6
    print(f"recommender.onnx: {N} fragrances, {A} accords, {size:.1f} MB -> {OUT.resolve()}")
    return traits, accords, items, E16.astype(np.float32), POP, count, P, D16.astype(np.float32)


def verify(traits, accords, items, E, POP, count, P, dom):
    """Check the ONNX model against a plain-numpy implementation of the same rules."""
    sess = ort.InferenceSession(str(OUT / "recommender.onnx"), providers=["CPUExecutionProvider"])
    W1, b1, W2, b2 = load_mlp()
    G = np.array([GENDERS.index(it["gender"]) for it in items])
    rng = np.random.default_rng(1)
    for case in range(6):
        t = rng.dirichlet(np.ones(len(traits)) * .5).astype(np.float32)[None]
        if case % 2:
            t = t * 3.7          # un-normalised input must give identical results (model normalises)
        t_ref = t / t.sum()
        allowed = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]][case % 4], np.float32)[None]
        unique = np.array([float(case == 5)], np.float32)
        pref = np.zeros((1, len(accords)), np.float32); must = pref.copy(); avoid = pref.copy()
        if case >= 2: pref[0, accords.index("Spicy")] = 1.5; must[0, accords.index("Spicy")] = 1
        if case >= 3: avoid[0, accords.index("Sweet")] = 1
        out = sess.run(None, {"traits": t, "allowed": allowed, "unique": unique,
                              "accord_pref": pref, "must": must, "avoid": avoid})
        # numpy reference
        q = np.maximum(t_ref @ W1 + b1, 0) @ W2 + b2; q /= np.linalg.norm(q)
        match = E @ q[0] + ACCORD_WEIGHT * (dom @ pref[0])
        ok = allowed[0][G] > 0
        ok &= (count < UNIQUE_MAX) if unique[0] else (count >= POPULAR_MIN)
        ok &= (P @ must[0] == must[0].sum()) & (P @ avoid[0] == 0)
        if not unique[0]:
            ok &= match >= match[ok].max() - RELEVANCE_WINDOW
        final = match + (0 if unique[0] else POP_WEIGHT * POP)
        scored = np.where(ok, final, -np.inf)
        ref = scored.argsort()[::-1][:5]
        got = out[0][:5]
        # Same items in the same order, OR any swap is a float32 near-tie (score gap < 1e-5).
        same = list(got) == list(ref) or (set(got) == set(ref) and np.abs(scored[got] - scored[ref]).max() < 1e-5)
        print(f"case {case}: onnx {got.tolist()}  numpy {ref.tolist()}  "
              f"max score diff {np.abs(out[1][:5] - scored[got]).max():.1e}  {'OK' if same else 'MISMATCH'}")


if __name__ == "__main__":
    verify(*build())
