"""Scentmate API — serves model/recommender.onnx on a Hugging Face Space (CPU).

Endpoints
  GET  /                 health + model info (also wakes a sleeping Space)
  GET  /meta             traits, accords and note vocabulary the frontend needs
  POST /recommend        personality + preferences -> ranked fragrances      (rate limited per IP)
  POST /feedback         "I like this" -> appended to feedback.jsonl for retraining (rate limited)
  GET  /feedback/export  download feedback.jsonl (needs header X-Admin-Token = ADMIN_TOKEN secret)

Environment variables (Space → Settings → Variables and secrets)
  ALLOWED_ORIGINS     comma-separated, e.g. https://scentmate.netlify.app   (default "*")
  RATE_LIMIT_PER_DAY  recommendations per visitor per UTC day                (default 150)
  ADMIN_TOKEN         secret for /feedback/export                            (export disabled if unset)
  IP_SALT             secret salt so stored IP hashes can't be reversed       (random per boot if unset)
"""
import hashlib, json, os, secrets, sqlite3, threading, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
DATA_DIR = HERE / "data"                       # NOTE: free Spaces lose this folder on restart
DATA_DIR.mkdir(exist_ok=True)

RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_DAY", "150"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
IP_SALT = os.getenv("IP_SALT") or secrets.token_hex(16)
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# ---------------------------------------------------------------- model
CONFIG = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
TRAITS, ACCORDS = CONFIG["traits"], CONFIG["accords"]
ITEMS = json.loads((MODEL_DIR / "items.json").read_text(encoding="utf-8"))   # [name, brand, accords, notes, g, count]
NOTES_LOWER = [[n.lower() for n in it[3]] for it in ITEMS]
GENDER_CODE = {"w": "women", "m": "men", "u": "unisex"}
ALLOWED = {"women": [1, 0, 1], "men": [0, 1, 1], "unisex": [0, 0, 1], None: [1, 1, 1]}

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2                  # free Spaces have 2 vCPUs
SESSION = ort.InferenceSession(str(MODEL_DIR / "recommender.onnx"), opts, providers=["CPUExecutionProvider"])


def run_model(traits, gender, unique, accord_pref, must, avoid):
    vec = lambda d: np.array([[float(d.get(a, 0)) for a in ACCORDS]], np.float32)
    feeds = {
        "traits": np.array([[max(0.0, float(traits.get(t, 0))) for t in TRAITS]], np.float32),
        "allowed": np.array([ALLOWED[gender]], np.float32),
        "unique": np.array([1.0 if unique else 0.0], np.float32),
        "accord_pref": vec(accord_pref),
        "must": vec({a: 1 for a in must}),
        "avoid": vec({a: 1 for a in avoid}),
    }
    if not feeds["traits"].any():              # no personality info yet -> neutral blend
        feeds["traits"][:] = 1.0
    ids, scores, matches, affinity, n_cons, n_ranked = SESSION.run(None, feeds)
    return ids, scores, matches, affinity, int(n_cons[0]), int(n_ranked[0]), feeds["traits"][0]


# ---------------------------------------------------------------- rate limiting (SQLite, per hashed IP)
_db_lock = threading.Lock()
_db = sqlite3.connect(DATA_DIR / "usage.db", check_same_thread=False, isolation_level=None)
_db.execute("CREATE TABLE IF NOT EXISTS usage(who TEXT, day TEXT, count INTEGER, PRIMARY KEY(who, day))")


def client_id(request: Request, bucket: str) -> str:
    # Spaces sit behind a proxy: the real client is the first X-Forwarded-For entry.
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    return bucket + ":" + hashlib.sha256((IP_SALT + ip).encode()).hexdigest()[:32]


def consume(who: str) -> tuple[bool, int]:
    """Atomically use one request from today's quota. Returns (allowed, remaining)."""
    day = datetime.now(timezone.utc).date().isoformat()
    with _db_lock:
        row = _db.execute("""INSERT INTO usage VALUES (?, ?, 1)
                             ON CONFLICT(who, day) DO UPDATE SET count = count + 1 WHERE count < ?
                             RETURNING count""", (who, day, RATE_LIMIT)).fetchone()
    return (row is not None, RATE_LIMIT - row[0] if row else 0)


def seconds_to_reset() -> int:
    now = datetime.now(timezone.utc)
    return int(((now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) - now).total_seconds())


def rate_limit(request: Request, bucket: str) -> int:
    ok, remaining = consume(client_id(request, bucket))
    if not ok:
        raise HTTPException(429, "Daily limit reached — come back tomorrow!",
                            headers={"Retry-After": str(seconds_to_reset())})
    return remaining


# ---------------------------------------------------------------- API
app = FastAPI(title="Scentmate API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["X-RateLimit-Remaining", "Retry-After"])

AccordName = str


class Constraint(BaseModel):
    type: Literal["must", "avoid"]
    accord: AccordName


class RecommendIn(BaseModel):
    traits: dict[str, float] = Field(default_factory=dict, examples=[{"romantic": 1.2, "mysterious": 0.5}])
    gender: Literal["women", "men", "unisex"] | None = None
    unique: bool = False
    accord_prefs: dict[str, float] = Field(default_factory=dict, examples=[{"Spicy": 1.5, "Sweet": -1}])
    constraints: list[Constraint] = Field(default_factory=list, max_length=20)   # oldest first
    notes: dict[str, float] = Field(default_factory=dict, max_length=30)          # {"vanilla": 1, "rose": -1}
    exclude: list[int] = Field(default_factory=list, max_length=300)              # ids already shown
    top_k: int = Field(3, ge=1, le=10)


class FeedbackIn(BaseModel):
    traits: dict[str, float]
    gender: Literal["women", "men", "unisex"] | None = None
    item_id: int


@app.get("/")
def health():
    return {"status": "ok", "model": "recommender.onnx", "fragrances": len(ITEMS), "traits": TRAITS}


@app.get("/meta")
def meta():
    return {"traits": TRAITS, "accords": ACCORDS, "notes": CONFIG["notes"], "rate_limit_per_day": RATE_LIMIT}


@app.post("/recommend")
def recommend(body: RecommendIn, request: Request):
    bad = [t for t in body.traits if t not in TRAITS] + \
          [a for a in list(body.accord_prefs) + [c.accord for c in body.constraints] if a not in ACCORDS]
    if bad:
        raise HTTPException(422, f"unknown traits/accords: {sorted(set(bad))}")
    remaining = rate_limit(request, "rec")

    # Hard constraints: if fewer than top_k fragrances satisfy them all, drop the OLDEST and retry,
    # so the user's most recent request always wins. The model reports how many items qualify.
    cons = list(body.constraints)
    while True:
        must = [c.accord for c in cons if c.type == "must"]
        avoid = [c.accord for c in cons if c.type == "avoid"]
        ids, scores, matches, aff, n_cons, n_ranked, tvec = run_model(
            body.traits, body.gender, body.unique, body.accord_prefs, must, avoid)
        if n_cons >= body.top_k or not cons:
            break
        cons.pop(0)
    relaxed = [c.model_dump() for c in body.constraints if c not in cons]

    # Candidates = every fragrance that satisfies the filters (up to the model's top-K). The first
    # n_ranked are the best-fitting ones; the rest are demoted by the model (score -100) and only
    # appear once the best ones have all been shown ("show me others").
    n = max(1, min(n_cons, len(ids)))
    notes = {k.lower(): v for k, v in body.notes.items()}
    cands = []
    for j in range(n):
        i = int(ids[j])
        # Re-rank with note preferences (e.g. "with vanilla", "no rose").
        nb = sum(0.1 * v for note, v in notes.items() if note in NOTES_LOWER[i])
        cands.append((float(scores[j]) + nb, float(matches[j]) + nb, i, aff[j]))
    cands.sort(key=lambda c: -c[0])

    # Skip fragrances the user has already seen; when everything has been shown, start over.
    excl = set(body.exclude)
    fresh = [c for c in cands if c[2] not in excl]
    reset = len(fresh) < body.top_k and len(cands) >= body.top_k
    picked = (cands if reset else fresh)[: body.top_k]

    results = []
    for score, match, i, a in picked:
        name, brand, accords, notes_, g, count = ITEMS[i]
        results.append({
            "id": i, "name": name, "brand": brand, "accords": accords, "notes": notes_,
            "gender": GENDER_CODE[g], "ratingCount": count,
            "match": round(match, 4), "score": round(score, 4),
            "topTrait": TRAITS[int(np.argmax(tvec * a))],        # strongest weighted trait for this pick
        })
    return {"results": results, "relaxed": relaxed, "reset_shown": reset, "remaining_today": remaining}


@app.post("/feedback")
def feedback(body: FeedbackIn, request: Request):
    if not 0 <= body.item_id < len(ITEMS):
        raise HTTPException(422, "unknown item_id")
    rate_limit(request, "fb")
    row = {"traits": {t: float(v) for t, v in body.traits.items() if t in TRAITS},
           "gender": body.gender, "liked_index": body.item_id, "ts": int(time.time())}
    with _db_lock, open(DATA_DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return {"saved": True}


@app.get("/feedback/export")
def export_feedback(x_admin_token: str = Header("")):
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(403, "forbidden")
    path = DATA_DIR / "feedback.jsonl"
    if not path.exists():
        raise HTTPException(404, "no feedback yet")
    return FileResponse(path, media_type="application/x-ndjson", filename="feedback.jsonl")
