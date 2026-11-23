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
| `train.py` | Preprocess the dataset, learn the 64-d scent space, train the trait encoder |
| `enrich.py` | Add `gender` (name rules + classifier) and `popularity` columns to the dataset |
| `build_model.py` | Fuse encoder + catalogue + ranking rules into `site/model/` and verify it |
| `data/` | Put `fragrances_clean.json` here (not committed) |

## Run locally

Browsers refuse to load the model from `file://`, so serve the folder:

```bash
python -m http.server 8080 -d site
```
Open http://localhost:8080. (The feedback button needs Netlify; use `npx netlify dev` to test it.)

## Rebuild the model

```bash
pip install -r requirements.txt
python enrich.py        # only when the dataset changes
python train.py         # -> artifacts/
python build_model.py   # -> site/model/   (then commit + push; Netlify redeploys)
```

### Learning from real users
1. Set `ADMIN_TOKEN` in Netlify → Site configuration → Environment variables.
2. Download the likes:
   `curl -H "X-Admin-Token: <token>" https://<your-site>.netlify.app/api/feedback > feedback.jsonl`
3. Run `train.py` and `build_model.py` again (with the same dataset file), commit, push.
