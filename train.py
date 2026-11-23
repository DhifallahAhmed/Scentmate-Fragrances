"""train.py - preprocess fragrances, build embeddings, train + export the trait model."""
import gzip, json, re, sys, unicodedata
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
import onnx
from onnx import helper, numpy_helper, TensorProto

HERE = Path(__file__).parent
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "data" / "fragrances_clean.json")
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
EMB_DIM = 64

# --- 1. Feature weighting: repeat tokens so accords/heart/base count more than fleeting top notes.
TIER_REPEAT = {"mainAccords": 3, "topNotes": 1, "middleNotes": 2, "baseNotes": 2}
PREFIX = {"mainAccords": "acc:", "topNotes": "note:", "middleNotes": "note:", "baseNotes": "note:"}
JUNK = {"acc:main_accords"}  # header row that leaked into the data

def norm(s: str) -> str:
    """'Pink pepper' -> 'pink_pepper', 'Fougère' -> 'fougère' (NFKC + casefold)."""
    s = unicodedata.normalize("NFKC", s).casefold().strip()
    return re.sub(r"[^\w]+", "_", s).strip("_")

def tokens(rec: dict) -> list[str]:
    out = []
    for field, rep in TIER_REPEAT.items():
        for item in rec.get(field) or []:          # missing/None -> empty list
            tok = PREFIX[field] + norm(item)
            if tok not in JUNK:
                out.extend([tok] * rep)
    return out

# --- 2. Trait lexicon: the "labels" this dataset lacks. Hand-authored, tune freely.
TRAITS = {
    "adventurous":  {"acc:spicy": 1.0, "acc:smoky": .8, "acc:leathery": .7, "acc:earthy": .6, "acc:aquatic": .4, "note:pink_pepper": .6, "note:vetiver": .5},
    "romantic":     {"acc:floral": 1.0, "acc:sweet": .6, "acc:powdery": .6, "note:rose": .8, "note:jasmine": .6, "note:vanilla": .5},
    "professional": {"acc:fresh": .9, "acc:woody": .8, "acc:citrus": .6, "acc:green": .4, "note:bergamot": .6, "note:iris": .5, "note:vetiver": .5},
    "playful":      {"acc:fruity": 1.0, "acc:gourmand": .8, "acc:sweet": .7, "acc:citrus": .4, "note:raspberry": .5},
    "mysterious":   {"acc:oriental": 1.0, "acc:resinous": .8, "acc:smoky": .7, "acc:animal": .5, "note:incense": .7, "note:oud": .7},
    "calm":         {"acc:green": .8, "acc:aquatic": .7, "acc:powdery": .4, "note:lavender": .8, "note:green_tea": .6},
    "confident":    {"acc:leathery": .8, "acc:woody": .8, "acc:chypre": .7, "acc:fougère": .6, "note:patchouli": .6, "note:tobacco": .5},
    "cozy":         {"acc:gourmand": .9, "acc:creamy": .8, "acc:sweet": .5, "note:vanilla": .8, "note:tonka_bean": .7, "note:amber": .5},
}
TRAIT_NAMES = list(TRAITS)

def mlp_to_onnx(mlp: MLPRegressor, n_in: int) -> onnx.ModelProto:
    """Write the trained MLP as an explicit ONNX graph: Gemm -> Relu -> Gemm.
    (skl2onnx flattens multi-output MLPRegressor to (N*64, 1), so we build it ourselves.)"""
    W1, W2 = (w.astype(np.float32) for w in mlp.coefs_)
    b1, b2 = (b.astype(np.float32) for b in mlp.intercepts_)
    inits = [numpy_helper.from_array(a, n) for a, n in [(W1, "W1"), (b1, "b1"), (W2, "W2"), (b2, "b2")]]
    nodes = [helper.make_node("Gemm", ["traits", "W1", "b1"], ["h"]),
             helper.make_node("Relu", ["h"], ["h_relu"]),
             helper.make_node("Gemm", ["h_relu", "W2", "b2"], ["embedding"])]
    graph = helper.make_graph(
        nodes, "trait_encoder",
        [helper.make_tensor_value_info("traits", TensorProto.FLOAT, [None, n_in])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, W2.shape[1]])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9   # pin: newer `onnx` writes IR versions that older runtimes reject
    onnx.checker.check_model(model)
    return model

def key_hash(key: str) -> int:
    """32-bit FNV-1a of the stable page key. The browser gets this compact number per fragrance and sends
    it back with feedback; train.py maps it back to the fragrance (build_model.py checks there are no collisions)."""
    h = 0x811C9DC5
    for b in key.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h

def load_feedback(recs):
    """Real "I'd wear this" clicks (exported from /api/feedback) -> list of (traits, row index in recs).
    New rows carry key_hash (stable across rebuilds). Old rows carry liked_index = a position in the model
    that was live when they were collected; data/legacy_index_keys.json.gz translates those to keys."""
    fb = Path("feedback.jsonl")
    if not fb.exists():
        return []
    by_key = {r["key"]: i for i, r in enumerate(recs)}
    by_hash = {key_hash(k): i for k, i in by_key.items()}
    legacy_path = HERE / "data" / "legacy_index_keys.json.gz"
    legacy = json.load(gzip.open(legacy_path, "rt", encoding="utf-8")) if legacy_path.exists() else []
    out, skipped = [], 0
    for line in fb.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "key_hash" in row:
            i = by_hash.get(row["key_hash"])
        elif "liked_index" in row and 0 <= row["liked_index"] < len(legacy):
            i = by_key.get(legacy[row["liked_index"]])
        else:
            i = None
        if i is None or sum(row.get("traits", {}).values()) <= 0:
            skipped += 1                       # fragrance no longer recommended, or empty profile
            continue
        out.append((row["traits"], i))
    print(f"feedback: {len(out)} usable rows, {skipped} skipped")
    return out

def main():
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    # Recommend real perfumes only (no after-shaves / hair mists — see clean.py productType), and drop
    # records with zero olfactory info (~29%): they can never be matched.
    recs = [r for r in raw if r.get("productType", "fragrance") == "fragrance" and tokens(r)]
    print(f"kept {len(recs)}/{len(raw)} fragrances with notes/accords")

    # --- 3. Sparse TF-IDF over accord+note tokens (rare notes < 5 uses dropped as noise).
    vec = TfidfVectorizer(analyzer=tokens, min_df=5, dtype=np.float32)
    X = vec.fit_transform(recs)                       # (n_fragrances, vocab) sparse
    # Records whose every token is too rare to be in the vocabulary get an all-zero fingerprint and
    # could never match anyone — drop them.
    nonzero = np.asarray(X.getnnz(axis=1) > 0)
    if not nonzero.all():
        print(f"dropped {int((~nonzero).sum())} records with an empty fingerprint")
        recs, X = [r for r, k in zip(recs, nonzero) if k], X[nonzero]
    vocab = vec.vocabulary_
    print("vocab size", len(vocab))

    # --- 4. Compress to a dense 64-d "scent space" (LSA). Captures co-occurrence (rose~oud~saffron).
    svd = TruncatedSVD(n_components=EMB_DIM, random_state=0)
    E = normalize(svd.fit_transform(X)).astype(np.float32)   # unit rows -> dot == cosine

    # --- 5. Project each trait into the same scent space.
    T = sparse.lil_matrix((len(TRAIT_NAMES), len(vocab)), dtype=np.float32)
    for i, t in enumerate(TRAIT_NAMES):
        for tok, w in TRAITS[t].items():
            if tok in vocab:
                T[i, vocab[tok]] = w * vec.idf_[vocab[tok]]
            else:
                print(f"  warn: lexicon token {tok!r} not in vocab")
    trait_emb = svd.transform(T.tocsr())              # (n_traits, 64), un-normalised on purpose

    # --- 6. Training data: synthetic trait mixes (+ real feedback later, see feedback.jsonl).
    rng = np.random.default_rng(0)
    Xs = rng.dirichlet(np.full(len(TRAIT_NAMES), 0.3), size=20000).astype(np.float32)
    Ys = normalize(Xs @ trait_emb)
    # Real "I'd wear this" clicks (GET /api/feedback on the Netlify site, saved as feedback.jsonl).
    rows = load_feedback(recs)
    if rows:
        Xf = np.array([[max(0.0, t.get(n, 0.0)) for n in TRAIT_NAMES] for t, _ in rows], np.float32)
        Xf /= Xf.sum(axis=1, keepdims=True)           # same scale as training mixes (the model normalises too)
        Yf = E[[i for _, i in rows]]
        Xs, Ys = np.vstack([Xs, np.repeat(Xf, 20, 0)]), np.vstack([Ys, np.repeat(Yf, 20, 0)])  # upweight real data

    # --- 7. Tiny MLP: traits(8) -> 32 -> 64. Trains in seconds on CPU.
    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, early_stopping=True, random_state=0)
    mlp.fit(Xs, Ys)
    print("MLP R^2 on train:", round(mlp.score(Xs, Ys), 3))

    # --- 8. Export: ONNX model + embedding matrix + metadata.
    onnx.save(mlp_to_onnx(mlp, len(TRAIT_NAMES)), OUT / "trait_encoder.onnx")
    np.save(OUT / "fragrance_emb.npy", E)
    # gender / popularity come from enrich.py (run it once before training).
    meta = {"traits": TRAIT_NAMES,
            "items": [{"key": r["key"], "name": r["name"], "brand": r["brand"], "accords": r["mainAccords"],
                       "notes": list(dict.fromkeys(r["topNotes"] + r["middleNotes"] + r["baseNotes"]))[:8],
                       "gender": r.get("gender"), "popularity": r.get("popularity", 0.0),
                       # best available count: real (dataset / web) or, if unknown, the estimate (see enrich.py)
                       "ratingCount": r["ratingCount"] or r.get("ratingCountEstimated", 0),
                       "ratingSource": r.get("ratingSource", "dataset")} for r in recs]}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print("exported to", OUT.resolve())

if __name__ == "__main__":
    main()
