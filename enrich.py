"""enrich.py - add `gender`, `genderSource`, `popularity` and `popularityTier` columns to the JSON.

Usage:  python enrich.py [path\to\fragrances_clean.json]
A one-time backup (fragrances_clean.backup.json) is written next to the file before it is changed.

gender:
  1. Taken from the NAME when it says so ("pour Homme", "for Women", "Femme", "Uomo", ...)  -> genderSource "name"
  2. Otherwise predicted by a logistic-regression classifier trained on those name-labelled
     fragrances, using accords + notes as features.                                         -> genderSource "model"
     P(women) >= 0.65 -> "women", <= 0.35 -> "men", in between -> "unisex".
  3. Records with no accords/notes -> null.
popularity: 0-100, log-scaled ratingCount (log keeps a 2,700-rating giant from dwarfing everything).
"""
import json, re, shutil, sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from train import tokens   # same accord/note tokenisation as the recommender

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "fragrances_clean.json")

MEN   = re.compile(r"\b(pour homme|homme|for men|men|man|him|uomo|masculin\w*|hombre|herren|gentleman|boy|mr)\b", re.I)
WOMEN = re.compile(r"\b(pour femme|femme|for women|women|woman|her|donna|f[eé]minin\w*|mujer|lady|girl|miss|she|damen)\b", re.I)

def gender_from_name(name: str):
    m, w = bool(MEN.search(name)), bool(WOMEN.search(name))
    return "men" if m and not w else "women" if w and not m else None

def tier(count: int) -> str:
    return ("very popular" if count >= 500 else "popular" if count >= 100
            else "known" if count >= 20 else "niche")

def main():
    backup = DATA.with_name(DATA.stem + ".backup.json")
    if not backup.exists():
        shutil.copy2(DATA, backup)
        print("backup written:", backup)

    recs = json.loads(DATA.read_text(encoding="utf-8"))
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

    # --- Popularity
    max_log = np.log1p(max(r["ratingCount"] for r in recs))

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
        r["popularity"] = round(100 * float(np.log1p(r["ratingCount"]) / max_log), 1)
        r["popularityTier"] = tier(r["ratingCount"])

    # Write atomically: temp file first, then replace.
    tmp = DATA.with_suffix(".tmp")
    tmp.write_text(json.dumps(recs, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(DATA)

    from collections import Counter
    print("gender source:", counts)
    print("gender:", Counter(r["gender"] for r in recs))
    print("tiers:", Counter(r["popularityTier"] for r in recs))
    print("updated:", DATA)

if __name__ == "__main__":
    main()
