"""evaluate.py - score the SHIPPED model (site/model/) against hand-written personas (eval/personas.json).

Usage:  python evaluate.py                    print the report; exit 1 on a regression or a broken rule
        python evaluate.py --update-baseline  accept the current scores as the new baseline
build_model.py runs this automatically after building.

How a persona is scored — exactly like the browser does it:
  quiz answers (read from site/index.html, so the page stays the single source of truth)
  -> trait weights + accord preferences -> optional refinement ("fresher" = must Fresh, "less sweet" = avoid Sweet)
  -> recommender.onnx -> relax the oldest wish if nothing satisfies all -> top 10.

Quality (compared with eval/baseline.json; a drop beyond TOLERANCE fails):
  hit@3    share of personas with at least one acceptable fragrance in the first 3 shown
  hit@10   ... within the first 10 (i.e. after pressing "Show others" up to 3 times)
  recall@10  share of each persona's acceptable fragrances found in the top 10 (averaged)
  coverage   distinct fragrances in all top-3s / (3 x personas)  — low = everyone gets the same picks
Hard rules (any failure fails the build): results respect the chosen gender, "unique" results have
< 50 ratings, and every "must"/"avoid" wish is honoured whenever the model reports it is satisfiable.
"""
import json, re, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
MODEL = HERE / "site" / "model"
PERSONAS = HERE / "eval" / "personas.json"
BASELINE = HERE / "eval" / "baseline.json"
REPORT = HERE / "eval" / "report.json"
TOLERANCE = 0.05          # allowed drop in any quality score before the build fails
TOP_K = 10
ALLOWED = {"women": [1, 0, 1], "men": [0, 1, 1], "unisex": [0, 0, 1], None: [1, 1, 1]}
GENDER = {"w": "women", "m": "men", "u": "unisex"}
UNIQUE_MAX = 50


def parse_quiz(html: str):
    """Read the QUIZ array from site/index.html: per question, per option -> label, gender, traits, accords."""
    block = html[html.index("const QUIZ = ["): html.index("\n];", html.index("const QUIZ = ["))]
    num = lambda v: float(v)
    def obj(text, field):
        m = re.search(field + r":\s*\{([^}]*)\}", text)
        return {k.strip('"'): num(v) for k, v in re.findall(r'("?[\wÀ-ɏ]+"?)\s*:\s*(-?\d*\.?\d+)', m.group(1))} if m else {}
    questions = []
    for q in re.split(r"\n\s*\{\s*\n\s*q:", block)[1:]:
        opts = []
        for line in re.findall(r"\{ label:.*\}", q):
            g = re.search(r'gender:\s*("(\w+)"|null)', line)
            opts.append({"label": re.search(r'label:\s*"([^"]+)"', line).group(1),
                         "gender": (g.group(2) if g and g.group(2) else None) if g else "n/a",
                         "traits": obj(line, "traits"), "accords": obj(line, "accords")})
        questions.append(opts)
    assert len(questions) >= 6 and all(questions), "could not parse the quiz from index.html"
    return questions


class Recommender:
    """Python twin of recommend() in site/index.html."""
    def __init__(self):
        self.cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
        self.items = json.loads((MODEL / "items.json").read_text(encoding="utf-8"))
        self.sess = ort.InferenceSession(str(MODEL / "recommender.onnx"), providers=["CPUExecutionProvider"])
        self.T, self.A = self.cfg["traits"], self.cfg["accords"]

    def run(self, w, gender, unique, prefs, must, avoid):
        t = np.array([[max(0.0, w.get(k, 0.0)) for k in self.T]], np.float32)
        if not t.any():
            t[:] = 1
        v = lambda d: np.array([[d.get(a, 0.0) for a in self.A]], np.float32)
        ids, scores, matches, aff, n_cons, n_ranked = self.sess.run(None, {
            "traits": t, "allowed": np.array([ALLOWED[gender]], np.float32),
            "unique": np.array([1.0 if unique else 0.0], np.float32), "accord_pref": v(prefs),
            "must": v({a: 1 for a in must}), "avoid": v({a: 1 for a in avoid})})
        return [int(i) for i in ids], int(n_cons[0])

    def recommend(self, w, gender, unique, prefs, constraints, k=TOP_K):
        cons = list(constraints)
        while True:
            ids, n_cons = self.run(w, gender, unique, prefs,
                                   [a for typ, a in cons if typ == "must"], [a for typ, a in cons if typ == "avoid"])
            if n_cons >= 3 or not cons:
                break
            cons = cons[1:]                                  # drop the oldest wish, like the page does
        return ids[:max(1, min(n_cons, len(ids)))][:k], cons, n_cons


def resolve(items, brand, name):
    """[brand, exact name or unique prefix] -> item index. Raises if missing or ambiguous."""
    exact = [i for i, it in enumerate(items) if it[1] == brand and it[0] == name]
    if exact:
        return exact[0]
    pref = [i for i, it in enumerate(items) if it[1] == brand and it[0].startswith(name)]
    if len(pref) != 1:
        raise ValueError(f"expected fragrance {'not found' if not pref else 'is ambiguous'}: {brand} | {name}"
                         + (f" -> {[items[i][0] for i in pref[:4]]}" if pref else ""))
    return pref[0]


def persona_inputs(p, quiz):
    """Quiz answers -> (trait weights, accord prefs, gender) exactly as the chat accumulates them."""
    w, prefs, gender = {}, {}, None
    for q, label in zip(quiz, p["answers"]):
        opt = next((o for o in q if o["label"] == label), None)
        if opt is None:
            raise ValueError(f"{p['id']}: no answer {label!r} in question {[o['label'] for o in q]}")
        if opt["gender"] != "n/a":
            gender = opt["gender"]
        for t, x in opt["traits"].items():
            w[t] = w.get(t, 0) + x
        for a, x in opt["accords"].items():
            prefs[a] = max(-4, min(4, prefs.get(a, 0) + x))
    constraints = []
    for typ, sign in (("must", 1), ("avoid", -1)):
        for a in p.get("refine", {}).get(typ, []):
            prefs[a] = max(-4, min(4, prefs.get(a, 0) + 1.5 * sign))   # refinements count 1.5x, like the chat
            constraints.append((typ, a))
    return w, prefs, gender, constraints


def evaluate():
    quiz = parse_quiz((HERE / "site" / "index.html").read_text(encoding="utf-8"))
    rec = Recommender()
    personas = json.loads(PERSONAS.read_text(encoding="utf-8"))["personas"]
    rows, broken, top3_all = [], [], []
    for p in personas:
        w, prefs, gender, constraints = persona_inputs(p, quiz)
        expected = {resolve(rec.items, b, n) for b, n in p["expected"]}
        top, kept, n_cons = rec.recommend(w, gender, p.get("unique", False), prefs, constraints)
        top3_all += top[:3]
        # ---- hard rules
        for i in top:
            it = rec.items[i]
            if gender in ("women", "men") and GENDER[it[4]] not in (gender, "unisex"):
                broken.append(f"{p['id']}: gender rule broken by {it[0]}")
            if gender == "unisex" and GENDER[it[4]] != "unisex":
                broken.append(f"{p['id']}: non-unisex result {it[0]}")
            if p.get("unique") and it[5] >= UNIQUE_MAX:
                broken.append(f"{p['id']}: 'unique' returned a popular fragrance {it[0]} ({it[5]} ratings)")
            for typ, a in kept:                          # wishes still active after any relaxation
                if (typ == "must" and a not in it[2]) or (typ == "avoid" and a in it[2]):
                    broken.append(f"{p['id']}: '{typ} {a}' not honoured by {it[0]} {it[2]}")
        rows.append({"id": p["id"], "expected": len(expected),
                     "hit3": bool(expected & set(top[:3])) if expected else None,
                     "hit10": bool(expected & set(top)) if expected else None,
                     "recall10": len(expected & set(top)) / len(expected) if expected else None,
                     "first_hit": next((r + 1 for r, i in enumerate(top) if i in expected), None),
                     "top3": [rec.items[i][0] for i in top[:3]]})
    scored = [r for r in rows if r["expected"]]
    mean = lambda k: round(float(np.mean([r[k] for r in scored])), 3)
    summary = {"personas": len(rows), "scored": len(scored), "hit@3": mean("hit3"), "hit@10": mean("hit10"),
               "recall@10": mean("recall10"), "coverage": round(len(set(top3_all)) / max(1, len(top3_all)), 3)}
    return summary, rows, broken


def main():
    summary, rows, broken = evaluate()
    print(f"{'persona':24s} {'hit@3':>5} {'hit@10':>6} {'first':>5}  top 3")
    for r in rows:
        f = lambda v: "-" if v is None else ("yes" if v is True else "no" if v is False else str(v))
        print(f"{r['id']:24s} {f(r['hit3']):>5} {f(r['hit10']):>6} {f(r['first_hit']):>5}  {' | '.join(x[:24] for x in r['top3'])}")
    print("\nsummary:", summary)
    REPORT.write_text(json.dumps({"summary": summary, "personas": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    failed = False
    if broken:
        failed = True
        print("\nBROKEN RULES:"); [print("  " + b) for b in broken]
    if "--update-baseline" in sys.argv or not BASELINE.exists():
        verb = "updated" if BASELINE.exists() else "created"
        BASELINE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"baseline {verb}: {BASELINE}")
    else:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        for k in ("hit@3", "hit@10", "recall@10", "coverage"):
            if summary[k] < base[k] - TOLERANCE:
                failed = True
                print(f"REGRESSION: {k} {summary[k]} < baseline {base[k]} - {TOLERANCE}")
        if not failed:
            print("no regression vs baseline", {k: base[k] for k in ("hit@3", "hit@10", "recall@10", "coverage")})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
