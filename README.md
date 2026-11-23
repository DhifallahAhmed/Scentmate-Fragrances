# Scentmate — personality-based fragrance recommender

A chat that asks six questions about you and suggests fragrances whose notes fit your personality —
much-loved ones first, hidden gems on request — or finds fragrances similar to one you already love
("I love Dior Sauvage").

**Everything runs in the browser.** The trained ONNX model is downloaded once (≈ 8 MB compressed on a
first visit, cached afterwards) and executed with ONNX Runtime Web in a background thread. There is no
model server; visitors' answers never leave their device. Netlify hosts the static site plus two small
functions: feedback ("I'd wear this") for retraining, and anonymous health logging.

```
site/
├── index.html            chat, quiz, text understanding, rendering
├── worker.js             loads + runs the model off the main thread; ranking & "similar to" search
├── sw.js                 offline support, instant repeat visits
├── model/config.json     traits, accords, ranking parameters, names of the files below
├── model/files/          recommender.<hash>.onnx (3.9 MB) + items.<hash>.json — cached forever
└── vendor/ort/1.31.1/    ONNX Runtime Web (WebAssembly)
netlify/functions/        feedback.mjs (/api/feedback), log.mjs (/api/log) — stored in Netlify Blobs
```

## Repository layout

| Path | Purpose |
|---|---|
| `clean.py` | Raw Parfumo CSV → `data/fragrances_clean.json`: stable keys, missing ratings as `null`, junk/copied rows handled, product types |
| `enrich.py` | Accords imputed from notes, gender (name rules + classifier), rating source (dataset / web / estimated), popularity |
| `train.py` | Learn the 64-d scent space and the personality encoder (10 traits) from the perfumes |
| `build_model.py` | Fuse encoder + catalogue + ranking into one compact ONNX model, verify it, then run the quality gate |
| `evaluate.py` | Score the shipped model on `eval/personas.json`; `--tune` searches the ranking parameters |
| `data/rating_overrides.csv` | Real rating counts for flagships the dataset is missing (looked up on the web, with sources) |
| `tools/size_budget.py` | Fails if a first visit would download more than 12 MB |
| `tools/check_deploy.py` | Checks the live site: compression, cache headers, transfer sizes |
| `tests/` | `functions.mjs` (Netlify functions) and `smoke.mjs` (real browser, CPU slowed 4×) |
| `tools/run_checks.py` | Runs every check above in one go — use it before pushing |

## Run locally

```bash
python -m http.server 8080 -d site          # open http://localhost:8080
```
(Browsers refuse to load the model from `file://`. Feedback/logging need Netlify: `npx netlify dev`.)

## Rebuild the model

```bash
pip install -r requirements.txt
python clean.py         # data/02_Parfumo_Perfumes.csv.gz -> data/fragrances_clean.json
python enrich.py        # accords, gender, ratings, popularity
python train.py         # -> artifacts/
python build_model.py   # -> site/model/, verified, then the quality gate (fails on a regression)
```
After an intended quality change: `python evaluate.py --update-baseline`.
To retune ranking: `python evaluate.py --tune` (writes `eval/params.json`), then `python build_model.py`.

## Checks (run before pushing)

```bash
python tools/run_checks.py                                     # quality gate, size budget, function + browser tests
python tools/check_deploy.py https://<your-site>.netlify.app   # after deploying
```
The browser test needs `cd tests && npm install` once (it uses Edge on Windows, Playwright's Chromium elsewhere).

## Operating it (Netlify → Site configuration → Environment variables)

| Variable | Purpose |
|---|---|
| `ADMIN_TOKEN` | Required to export feedback / logs |
| `IP_SALT` | Optional random string used when hashing visitor IPs for rate limits |

- Feedback: `curl -H "X-Admin-Token: <token>" https://<site>/api/feedback > feedback.jsonl`, then rebuild.
  Likes are stored per fragrance key (stable across rebuilds), de-duplicated per browser tab and capped.
- Health: `curl -H "X-Admin-Token: <token>" "https://<site>/api/log?day=YYYY-MM-DD"` — first line is a
  summary (loads, failure rate, median load time, errors).
