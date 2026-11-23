# Scentmate — personality-based fragrance recommender

A chat that asks six questions about you and suggests fragrances whose notes fit your personality —
much-loved ones first, hidden gems on request.

**Everything runs in the browser.** The trained ONNX model (`site/model/recommender.onnx`) is
downloaded once and executed with ONNX Runtime Web — no model server, no cold starts, and the
user's answers never leave their device. The whole app is a static Netlify site plus one tiny
function that stores "I'd wear this" feedback for retraining.

```
Netlify
├── site/                       static page
│   ├── index.html              chat, quiz, text understanding, ranking logic
│   ├── model/recommender.onnx  trained encoder + 41,704 fragrances + filters + ranking (8 MB)
│   └── vendor/ort/             ONNX Runtime Web (WebAssembly)
└── netlify/functions/feedback.mjs   POST/GET /api/feedback  (stored in Netlify Blobs)
```

## Repository layout

| Path | Purpose |
|---|---|
| `site/` | Everything Netlify publishes |
| `netlify/functions/feedback.mjs` | Saves likes; `GET` with `X-Admin-Token` exports them as JSONL |
| `clean.py` | Raw Parfumo CSV → `data/fragrances_clean.json`: stable keys, missing ratings as `null`, junk rows removed, product types |
| `enrich.py` | Add gender (name rules + classifier), rating source (dataset / web / estimated) and popularity |
| `train.py` | Learn the 64-d scent space and the trait encoder (perfumes only) |
| `build_model.py` | Fuse encoder + catalogue + ranking rules into `site/model/`, verify it, then run the evaluation |
| `evaluate.py` | Score the shipped model on `eval/personas.json`; fails on broken rules or a drop vs `eval/baseline.json` |
| `data/` | Raw CSV + generated dataset (not committed); `rating_overrides.csv` and `legacy_index_keys.json.gz` are committed |

## Run locally

Browsers refuse to load the model from `file://`, so serve the folder:

```bash
python -m http.server 8080 -d site
```
Open http://localhost:8080. (The feedback button needs Netlify; use `npx netlify dev` to test it.)

## Rebuild the model

```bash
pip install -r requirements.txt
python clean.py         # data/02_Parfumo_Perfumes.csv.gz -> data/fragrances_clean.json
python enrich.py        # gender, rating source, popularity
python train.py         # -> artifacts/
python build_model.py   # -> site/model/, then the quality gate (evaluate.py); commit + push if it passes
```

### Learning from real users
1. Set `ADMIN_TOKEN` in Netlify → Site configuration → Environment variables.
2. Download the likes:
   `curl -H "X-Admin-Token: <token>" https://<your-site>.netlify.app/api/feedback > feedback.jsonl`
3. Run `train.py` and `build_model.py` again, commit, push. Likes are stored against each fragrance's
   stable key, so they stay valid even when the dataset is re-cleaned (older position-based likes are
   translated with `data/legacy_index_keys.json.gz`).
