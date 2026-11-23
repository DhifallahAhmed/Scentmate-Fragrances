"""enrich.py - add gender, rating-source and popularity columns to the dataset.

Usage:  python enrich.py [path\to\fragrances_clean.json]
Run after clean.py (which rebuilds the file from the raw CSV, so enrich.py can be re-run any time).

gender:
  1. Taken from the NAME when it says so ("pour Homme", "for Women", "Femme", "Uomo", ...)  -> genderSource "name"
  2. Otherwise predicted by a logistic-regression classifier trained on those name-labelled
     fragrances, using accords + notes as features.                                         -> genderSource "model"
     P(women) >= 0.65 -> "women", <= 0.35 -> "men", in between -> "unisex".
  3. Records with no accords/notes -> null.

ratings — the Parfumo dataset is MISSING the rating count for ~49% of fragrances (clean.py stores
those as null; the source has no real zeros). The missing ones include the biggest
bestsellers (Sauvage EDT, Bleu de Chanel, Aventus...). So every record gets a source:
  ratingSource "dataset"    count came with the dataset
               "web"        count filled from data/rating_overrides.csv (looked up on the web)
               "estimated"  still unknown -> ratingCountEstimated is predicted by a model trained on
                            the known counts (line/brand popularity, year, notes...). Rough: good for
                            the long tail, it will underestimate unlisted flagships -> add them to the CSV.
popularity: 0-100, log-scaled count (real if known, else estimated). popularityTier from the same count.
"""
import collections, csv, json, re, sys, unicodedata
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from train import tokens   # same accord/note tokenisation as the recommender

HERE = Path(__file__).parent
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "data" / "fragrances_clean.json")
OVERRIDES = HERE / "data" / "rating_overrides.csv"

MEN   = re.compile(r"\b(pour homme|homme|for men|men|man|him|uomo|masculin\w*|hombre|herren|gentleman|boy|mr|m[aâ]le)\b", re.I)
WOMEN = re.compile(r"\b(pour femme|femme|for women|women|woman|her|donna|f[eé]minin\w*|mujer|lady|girl|miss|she|damen)\b", re.I)

def gender_from_name(name: str):
    m, w = bool(MEN.search(name)), bool(WOMEN.search(name))
    return "men" if m and not w else "women" if w and not m else None

def tier(count: float) -> str:
    return ("very popular" if count >= 500 else "popular" if count >= 100
            else "known" if count >= 20 else "niche")


# ------------------------------------------------------------------ ratings: web overrides
def apply_overrides(recs) -> int:
    """Fill missing counts from data/rating_overrides.csv. Matches on the stable `key` (falls back to exact
    brand + name). Never overwrites a count that came with the dataset — except a copied one (see clean.py)."""
    if not OVERRIDES.exists():
        return 0
    lines = [l for l in OVERRIDES.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    by_key = {r["key"]: r for r in recs}
    by_name = {(r["brand"], r["name"]): r for r in recs}
    applied = 0
    for row in csv.DictReader(lines):
        r = by_key.get(row.get("key") or "") or by_name.get((row["brand"], row["name"]))
        if r is None:
            print(f"  override not matched (check key/spelling): {row.get('key')} | {row['brand']} | {row['name']}")
            continue
        if not r["ratingCount"] or r.get("ratingSource") == "web":
            r["ratingCount"], r["ratingSource"] = int(row["ratingCount"]), "web"
            r.pop("ratingNote", None)
            applied += 1
    return applied


# ------------------------------------------------------------------ ratings: estimator for the rest
CONC = (r"(eau de toilette|eau de parfum|eau de cologne|extrait de parfum|extrait|parfum|perfume|cologne|elixir|intense|"
        r"absolu|absolue|edt|edp|lotion après-rasage|after ?shave|aftershave|deodorant|body (lotion|mist|spray)|"
        r"shower gel|hair mist|soap|spray)")
GENERIC = {"eau", "l'eau", "le", "la", "les", "the", "a", "un", "une", "my", "pour", "l'", "de", "du", "des", "acqua", "agua", "aqua"}

def _norm(s): return unicodedata.normalize("NFKC", s).casefold().strip()

def _base_name(r):
    n, b = _norm(r["name"]), _norm(r["brand"])
    m = re.match(rf"^(.*?)\s+{re.escape(b)}\s+\d{{4}}\b", n)        # "sauvage dior 2015 eau de toilette" -> "sauvage"
    if m and m.group(1):
        n = m.group(1)
    return re.sub(rf"\b{CONC}\b", " ", n).strip()

def _line_key(r):
    """The product line: brand + first meaningful word(s), e.g. 'dior|sauvage', 'chanel|bleu'."""
    toks = re.findall(r"[\w'’]+", _base_name(r))
    if not toks:
        return _norm(r["brand"]) + "|?"
    return _norm(r["brand"]) + "|" + " ".join(toks[:2] if toks[0] in GENERIC and len(toks) > 1 else toks[:1])

def estimate_missing(recs):
    """Predict log(ratings) for records without a count, from records with one. Returns (estimates, cv report)."""
    y = np.log1p([r["ratingCount"] or 0 for r in recs])
    known = np.array([bool(r["ratingCount"]) for r in recs])
    lines = [_line_key(r) for r in recs]
    brands = [_norm(r["brand"]) for r in recs]
    line_size, brand_size = collections.Counter(lines), collections.Counter(brands)

    def features(rows, train_rows):
        gl, gb = collections.defaultdict(list), collections.defaultdict(list)
        for i in train_rows:                                   # group stats only from the training rows (no leakage)
            gl[lines[i]].append((y[i], i)); gb[brands[i]].append((y[i], i))
        X = []
        for i in rows:
            r = recs[i]
            lv = [v for v, j in gl.get(lines[i], []) if j != i]
            bv = [v for v, j in gb.get(brands[i], []) if j != i]
            X.append([
                max(lv) if lv else np.nan, np.mean(lv) if lv else np.nan, len(lv), line_size[lines[i]],
                np.mean(bv) if bv else np.nan, np.percentile(bv, 90) if bv else np.nan, len(bv), brand_size[brands[i]],
                r["releaseYear"] or np.nan, len(r["topNotes"]) + len(r["middleNotes"]) + len(r["baseNotes"]),
                len(r["mainAccords"]), float(len(_base_name(r).split()) <= 2),
                float(bool(re.search(r"eau de (toilette|parfum)", _norm(r["name"])))),
            ])
        return np.array(X, float)

    model = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)
    kidx, uidx = np.where(known)[0], np.where(~known)[0]
    pred = np.zeros(len(kidx))                                  # honest 5-fold check on known counts
    for tr, te in KFold(5, shuffle=True, random_state=0).split(kidx):
        pred[te] = model().fit(features(kidx[tr], kidx[tr]), y[kidx[tr]]).predict(features(kidx[te], kidx[tr]))
    from scipy.stats import spearmanr
    report = (f"rank correlation {spearmanr(pred, y[kidx]).correlation:.2f}, "
              f"median error {np.median(np.abs(np.expm1(pred) - np.expm1(y[kidx]))):.0f} ratings")
    m = model().fit(features(kidx, kidx), y[kidx])
    est = np.clip(np.expm1(m.predict(features(uidx, kidx))), 1, None)
    return dict(zip(uidx.tolist(), est.tolist())), report


# ------------------------------------------------------------------ accords from notes
def impute_accords(recs):
    """~9,800 fragrances list notes but no accords. Learn notes -> accords (one classifier per accord) from
    fragrances that have both, check it on held-out ones, then fill the gaps (accordsSource = "imputed").
    Accords are ordered by confidence, because the first accords count most in the model."""
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score
    note_tokens = lambda r: [_norm(n) for n in r["topNotes"] + r["middleNotes"] + r["baseNotes"]]
    has_notes = [bool(r["topNotes"] or r["middleNotes"] or r["baseNotes"]) for r in recs]
    both = [i for i, r in enumerate(recs) if has_notes[i] and r["mainAccords"]]
    todo = [i for i, r in enumerate(recs) if has_notes[i] and not r["mainAccords"]]
    if not todo:
        return 0, "nothing to impute"
    vec = TfidfVectorizer(analyzer=note_tokens, min_df=3)
    X = vec.fit_transform([recs[i] for i in both])
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([[a for a in recs[i]["mainAccords"] if a != "Main Accords"] for i in both])
    clf = lambda: OneVsRestClassifier(LogisticRegression(max_iter=1000, C=4.0), n_jobs=-1)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    P = clf().fit(Xtr, Ytr).predict_proba(Xte)
    best = max(((f1_score(Yte, (P >= t).astype(int), average="micro"), t) for t in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)))
    report = f"held-out micro-F1 {best[0]:.2f} at threshold {best[1]}"
    model = clf().fit(X, Y)
    P = model.predict_proba(vec.transform([recs[i] for i in todo]))
    filled = 0
    for i, p in zip(todo, P):
        order = [k for k in np.argsort(-p) if p[k] >= best[1]][:5] or [int(np.argmax(p))]
        recs[i]["mainAccords"] = [mlb.classes_[k] for k in order]
        recs[i]["accordsSource"] = "imputed"
        filled += 1
    return filled, report


def main():
    recs = json.loads(DATA.read_text(encoding="utf-8"))
    filled, report = impute_accords(recs)
    print(f"accords imputed from notes: {filled} ({report})")
    for r in recs:                     # flag fragrances whose fingerprint rests on accords alone
        r["lowInfo"] = not (r["topNotes"] or r["middleNotes"] or r["baseNotes"])
    has_notes = [bool(tokens(r)) for r in recs]

    # --- Gender: name labels -> train classifier -> fill the rest
    name_lbl = [gender_from_name(r["name"]) if h else None for r, h in zip(recs, has_notes)]
    vec = TfidfVectorizer(analyzer=tokens, min_df=5)
    idx_all = [i for i, h in enumerate(has_notes) if h]
    X_all = vec.fit_transform([recs[i] for i in idx_all])
    pos = {i: k for k, i in enumerate(idx_all)}                      # record index -> matrix row

    lab = [i for i in idx_all if name_lbl[i]]
    X_lab = X_all[[pos[i] for i in lab]]
    y_lab = np.array([name_lbl[i] == "women" for i in lab], dtype=int)

    # Hold-out check so we know how much to trust the inferred labels.
    Xtr, Xte, ytr, yte = train_test_split(X_lab, y_lab, test_size=0.2, random_state=0, stratify=y_lab)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0).fit(Xtr, ytr)
    print(f"gender classifier hold-out accuracy: {clf.score(Xte, yte):.3f}  (n_test={len(yte)})")
    clf.fit(X_lab, y_lab)                                            # refit on everything labelled
    p_women = clf.predict_proba(X_all)[:, 1]

    counts = {"name": 0, "model": 0, None: 0}
    for i, r in enumerate(recs):
        if name_lbl[i]:
            r["gender"], r["genderSource"] = name_lbl[i], "name"
        elif has_notes[i]:
            p = p_women[pos[i]]
            r["gender"] = "women" if p >= 0.65 else "men" if p <= 0.35 else "unisex"
            r["genderSource"] = "model"
        else:
            r["gender"], r["genderSource"] = None, None
        counts[r["genderSource"]] += 1

    # --- Ratings: dataset -> web overrides -> estimate the rest
    for r in recs:
        if r.get("ratingSource") is None:
            r["ratingSource"] = "dataset" if r["ratingCount"] else "estimated"
    print("web overrides applied:", apply_overrides(recs))
    est, report = estimate_missing(recs)
    print("rating estimator (cross-validated on known counts):", report)
    for i, r in enumerate(recs):
        if i in est:
            r["ratingSource"], r["ratingCountEstimated"] = "estimated", int(round(est[i]))
        else:
            r.pop("ratingCountEstimated", None)

    # --- Popularity from the best available count
    effective = [r["ratingCount"] or r.get("ratingCountEstimated", 0) for r in recs]
    max_log = np.log1p(max(effective))
    for r, c in zip(recs, effective):
        r["popularity"] = round(100 * float(np.log1p(c) / max_log), 1)
        r["popularityTier"] = tier(c)

    # Write atomically: temp file first, then replace.
    tmp = DATA.with_suffix(".tmp")
    tmp.write_text(json.dumps(recs, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(DATA)

    print("gender source:", counts)
    print("gender:", collections.Counter(r["gender"] for r in recs))
    print("rating source:", collections.Counter(r["ratingSource"] for r in recs))
    print("tiers:", collections.Counter(r["popularityTier"] for r in recs))
    print("updated:", DATA)

if __name__ == "__main__":
    main()
