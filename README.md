# Scentmate — personality-based fragrance recommender

A chat that asks a few questions about your personality and recommends fragrances whose notes fit you —
popular ones first, hidden gems on request.

```
site/index.html  (Netlify)  ──HTTPS──►  hf_space/app.py  (Hugging Face Space, CPU)
chat + quiz + text parsing              └─ model/recommender.onnx
                                           trained trait encoder + 41,704 fragrances,
                                           filtering and ranking in one ONNX graph
```

## Repository layout

| Path | Purpose |
|---|---|
| `site/` | Frontend — static page deployed to Netlify (`index.html`, `_headers`) |
| `hf_space/` | Backend — FastAPI + ONNX Runtime, deployed as a Docker Space |
| `hf_space/model/` | Built model files served by the API |
| `train.py` | Preprocess the dataset, learn the 64-d scent space, train the trait encoder |
| `enrich.py` | Add `gender` (name rules + classifier) and `popularity` columns to the dataset |
| `build_model.py` | Fuse encoder + catalogue + ranking rules into `recommender.onnx` and verify it |
| `deploy_hf.py` | Create/update the Hugging Face Space |
| `data/` | Put `fragrances_clean.json` here (not committed) |

## Run locally

```bash
pip install -r requirements.txt -r hf_space/requirements.txt
cd hf_space && uvicorn app:app --port 7860
```
Open `site/index.html` in a browser — it talks to `http://localhost:7860` automatically.

## Rebuild the model

```bash
python enrich.py        # only when the dataset changes
python train.py         # -> artifacts/
python build_model.py   # -> hf_space/model/
python deploy_hf.py     # upload to the Space
```

Collected "♡ Love it" feedback can be exported from the Space (`GET /feedback/export` with the
`X-Admin-Token` header) and saved as `feedback.jsonl` here; `train.py` then learns from it.

## Configuration (Space → Settings → Variables and secrets)

| Name | Example | Meaning |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://scentmate.netlify.app` | Sites allowed to call the API |
| `RATE_LIMIT_PER_DAY` | `150` | Recommendations per visitor per day |
| `ADMIN_TOKEN` (secret) | random string | Protects `/feedback/export` |
| `IP_SALT` (secret) | random string | Salt for hashing visitor IPs |
