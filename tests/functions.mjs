// tests/functions.mjs — tests the Netlify Functions with an in-memory stand-in for Netlify Blobs.
// Usage: node tests/functions.mjs   (no install needed)
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = mkdtempSync(join(tmpdir(), "fn-"));
writeFileSync(join(dir, "blobs.mjs"), `
const stores = new Map();
export function getStore(name) {
  if (!stores.has(name)) stores.set(name, new Map());
  const m = stores.get(name);
  return {
    async get(k, o) { const v = m.get(k); return v === undefined ? null : (o?.type === "json" ? JSON.parse(v) : v); },
    async set(k, v) { m.set(k, String(v)); },
    async setJSON(k, v) { m.set(k, JSON.stringify(v)); },
    async list(o = {}) { return { blobs: [...m.keys()].filter(k => !o.prefix || k.startsWith(o.prefix)).map(key => ({ key })) }; },
  };
}`);
const load = async name => {
  const src = readFileSync(join(root, "netlify", "functions", name + ".mjs"), "utf8").replace('from "@netlify/blobs"', 'from "./blobs.mjs"');
  writeFileSync(join(dir, name + ".mjs"), src);
  return (await import(pathToFileURL(join(dir, name + ".mjs")).href)).default;
};

process.env.ADMIN_TOKEN = "test-token";
const feedback = await load("feedback"), log = await load("log");
const post = (fn, body, ip = "1.2.3.4") =>
  fn(new Request("http://x/api", { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body) }), { ip });
const sid = "11111111-2222-3333-4444-555555555555";

// feedback: save, de-duplicate, validate, cap, export
assert.equal((await post(feedback, { traits: { romantic: 1, evil: 9 }, gender: "women", key_hash: 7, sid })).status, 200);
assert.equal((await post(feedback, { traits: { romantic: 1 }, gender: "women", key_hash: 7, sid })).status, 200);
assert.equal((await post(feedback, { traits: {}, key_hash: 7, sid: "bad" })).status, 422);
assert.equal((await post(feedback, { traits: {}, key_hash: -1, sid })).status, 422);
let r; for (let i = 0; i < 31; i++) r = await post(feedback, { traits: { cozy: 1 }, key_hash: 100 + i, sid });
assert.equal(r.status, 429, "session cap");
const exported = await feedback(new Request("http://x/api", { headers: { "x-admin-token": "test-token" } }), {});
const rows = (await exported.text()).trim().split("\n").map(JSON.parse);
assert.equal(rows.length, 30, "30 unique likes in the session (duplicate click counted once)");
assert.ok(!("evil" in rows.find(x => x.key_hash === 7).traits), "unknown traits are dropped");
assert.equal((await feedback(new Request("http://x/api"), {})).status, 403, "export needs the token");

// log: accept, validate, summarise
assert.equal((await post(log, { type: "load", ok: true, ms: 900 })).status, 204);
assert.equal((await post(log, { type: "load", ok: false, msg: "x" })).status, 204);
assert.equal((await post(log, { type: "nope" })).status, 422);
assert.equal((await post(log, "x".repeat(5000))).status, 413);
const summary = JSON.parse((await (await log(new Request("http://x/api/log", { headers: { "x-admin-token": "test-token" } }), {})).text()).split("\n")[0]).summary;
assert.deepEqual([summary.loads, summary.failed, summary.loadMsMedian], [2, 1, 900]);

console.log("functions: all checks passed");
