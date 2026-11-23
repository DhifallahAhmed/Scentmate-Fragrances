"""tools/run_checks.py - run the same checks as the GitHub Action, on your own PC.

Usage:  python tools/run_checks.py            (from the project folder, with requirements installed)
Runs: model quality gate, download size budget, Netlify function tests, and the browser smoke test
(starts a local server for it). The smoke test needs `cd tests && npm install` once; on Windows it uses
the installed Microsoft Edge, elsewhere Playwright's Chromium (`npx playwright install chromium`).
Exit code 0 only if everything passes.
"""
import os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY, PORT = sys.executable, 8765

def run(name, cmd, **kw):
    print(f"\n=== {name} ===", flush=True)
    ok = subprocess.run(cmd, cwd=ROOT, **kw).returncode == 0
    print(f"--- {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok

results = {
    "quality gate": run("quality gate", [PY, "evaluate.py"]),
    "size budget": run("size budget", [PY, "tools/size_budget.py"]),
    "function tests": run("function tests", ["node", "tests/functions.mjs"]),
}

if not (ROOT / "tests" / "node_modules" / "playwright").exists():
    print("\n=== browser smoke test ===\nSKIPPED: run `cd tests && npm install` once to enable it.")
    results["browser smoke test"] = None
else:
    server = subprocess.Popen([PY, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "-d", "site"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/index.html", timeout=1); break
            except OSError:
                time.sleep(0.25)
        env = dict(os.environ, BASE_URL=f"http://localhost:{PORT}")
        if os.name == "nt" and "BROWSER_CHANNEL" not in env:
            env["BROWSER_CHANNEL"] = "msedge"          # use the Edge that ships with Windows
        results["browser smoke test"] = run("browser smoke test", ["node", "tests/smoke.mjs"], env=env)
    finally:
        server.terminate()

print("\n=== summary ===")
for name, ok in results.items():
    print(f"  {name:20s} {'PASS' if ok else 'SKIPPED' if ok is None else 'FAIL'}")
sys.exit(0 if all(ok is not False for ok in results.values()) else 1)
