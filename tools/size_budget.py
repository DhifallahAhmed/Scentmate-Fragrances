"""tools/size_budget.py - fail if a first visit would download too much.

Usage: python tools/size_budget.py
Counts every file a first-time visitor downloads, compressed the way a web server would (gzip level 6;
Netlify's brotli is a little smaller, so this is a safe upper bound).
"""
import gzip, json, sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
BUDGET_MB = 12.0            # total compressed first-visit download
MODEL_RAW_MB = 6.0          # the ONNX model file itself

cfg = json.loads((SITE / "model" / "config.json").read_text(encoding="utf-8"))
files = ["index.html", "worker.js", "sw.js", "favicon.svg", "model/config.json",
         "model/" + cfg["files"]["model"], "model/" + cfg["files"]["items"],
         *[str(p.relative_to(SITE)).replace("\\", "/") for p in (SITE / "vendor").rglob("*") if p.is_file() and p.suffix in (".js", ".mjs", ".wasm")]]
total = 0
print(f"{'file':52s} {'raw MB':>8} {'gzip MB':>8}")
for f in files:
    raw = (SITE / f).read_bytes()
    gz = len(gzip.compress(raw, 6))
    total += gz
    print(f"{f:52s} {len(raw) / 1e6:8.2f} {gz / 1e6:8.2f}")
model_raw = (SITE / "model" / cfg["files"]["model"]).stat().st_size / 1e6
print(f"\nfirst visit (compressed): {total / 1e6:.2f} MB  (budget {BUDGET_MB} MB)")
print(f"model file: {model_raw:.2f} MB  (budget {MODEL_RAW_MB} MB)")
fail = total / 1e6 > BUDGET_MB or model_raw > MODEL_RAW_MB
print("FAIL: over budget" if fail else "OK: within budget")
sys.exit(1 if fail else 0)
