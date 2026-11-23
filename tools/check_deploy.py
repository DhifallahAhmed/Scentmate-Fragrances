"""tools/check_deploy.py - check what your LIVE site actually sends: compression, cache headers, sizes.

Usage:  python tools/check_deploy.py https://your-site.netlify.app
Reports, for every file a first visit downloads, the Content-Encoding (br/gzip — or none!), the
Cache-Control header, and the bytes transferred. Exit code 1 if a big file is sent uncompressed
or a content-hashed file isn't cacheable.
"""
import json, sys, urllib.request

if len(sys.argv) != 2:
    sys.exit(__doc__)
BASE = sys.argv[1].rstrip("/")

def fetch(path, compressed=True):
    """compressed=True asks for br/gzip like a browser and returns the bytes actually transferred."""
    headers = {"User-Agent": "scentmate-deploy-check"}
    if compressed:
        headers["Accept-Encoding"] = "br, gzip"
    with urllib.request.urlopen(urllib.request.Request(BASE + "/" + path, headers=headers), timeout=60) as r:
        body = r.read()
        return r.headers, len(body), body

cfg = json.loads(fetch("model/config.json", compressed=False)[2])
files = ["index.html", "worker.js", "model/config.json", "model/" + cfg["files"]["model"], "model/" + cfg["files"]["items"],
         "vendor/ort/1.31.1/ort.wasm.min.js", "vendor/ort/1.31.1/ort-wasm-simd-threaded.mjs", "vendor/ort/1.31.1/ort-wasm-simd-threaded.wasm"]
problems, total = [], 0
print(f"{'file':50s} {'encoding':9s} {'sent MB':>8}  cache-control")
for f in files:
    h, n, _ = fetch(f)
    enc = h.get("Content-Encoding") or "none"
    cc = h.get("Cache-Control") or "-"
    total += n
    print(f"{f:50s} {enc:9s} {n / 1e6:8.2f}  {cc}")
    if enc == "none" and n > 200_000:
        problems.append(f"{f} is sent uncompressed ({n / 1e6:.1f} MB)")
    if ("/files/" in f or "/vendor/" in f) and "immutable" not in cc:
        problems.append(f"{f} is not cached as immutable")
print(f"\nfirst visit transferred: {total / 1e6:.2f} MB")
print("\n".join(["PROBLEMS:"] + ["  " + p for p in problems]) if problems else "OK: compressed and cacheable")
sys.exit(1 if problems else 0)
