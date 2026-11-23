// Netlify Function: stores "I'd wear this" clicks so the model can be retrained on real choices.
//   POST /api/feedback  {traits: {romantic: 1.2, ...}, gender: "women"|"men"|"unisex"|null, key_hash: 2166136261, sid: "<uuid>"}
//     key_hash = stable 32-bit id of the fragrance (site/model/files/items.*.json), unchanged across rebuilds
//     sid      = random id per browser tab (no personal data) — used to de-duplicate and cap floods
//   GET  /api/feedback  download everything as feedback.jsonl  (header X-Admin-Token = ADMIN_TOKEN)
// Data lives in Netlify Blobs. Environment variables: ADMIN_TOKEN (required for export), IP_SALT (optional).
import { getStore } from "@netlify/blobs";

const TRAITS = ["adventurous", "romantic", "professional", "playful", "mysterious", "calm", "confident", "cozy", "modern", "clean"];
const GENDERS = ["women", "men", "unisex"];
const MAX_PER_SESSION = 30;          // likes per browser tab
const MAX_PER_IP_PER_DAY = 100;      // likes per visitor per day (IP is only stored as a salted hash)

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("").slice(0, 24);
}

export default async (req, context) => {
  const store = getStore("feedback");

  if (req.method === "POST") {
    let body;
    try { body = await req.json(); } catch { return json({ detail: "invalid JSON" }, 400); }

    const keyHash = Number(body?.key_hash);
    if (!Number.isInteger(keyHash) || keyHash < 0 || keyHash > 0xFFFFFFFF) return json({ detail: "invalid key_hash" }, 422);
    const sid = typeof body?.sid === "string" && /^[0-9a-f-]{36}$/.test(body.sid) ? body.sid : null;
    if (!sid) return json({ detail: "invalid sid" }, 422);

    const day = new Date().toISOString().slice(0, 10);
    // Per-session cap. The key is session + fragrance, so clicking the same one twice overwrites, never adds.
    const { blobs: mine } = await store.list({ prefix: `fb/${sid}/` });
    if (mine.length >= MAX_PER_SESSION && !mine.some(b => b.key.endsWith(`/${keyHash}`))) return json({ detail: "limit reached" }, 429);
    // Per-IP daily cap (best effort: a counter blob per hashed IP and day).
    const ipKey = `rate/${day}/${await sha256((process.env.IP_SALT || "scentmate") + (context.ip || "unknown"))}`;
    const used = Number(await store.get(ipKey)) || 0;
    if (used >= MAX_PER_IP_PER_DAY) return json({ detail: "limit reached" }, 429);

    // Keep only known traits with sane values — never store arbitrary client data.
    const traits = {};
    for (const t of TRAITS) {
      const v = Number(body?.traits?.[t]);
      if (Number.isFinite(v) && v > 0) traits[t] = Math.min(v, 20);
    }
    const row = { traits, gender: GENDERS.includes(body?.gender) ? body.gender : null, key_hash: keyHash,
                  sid, ts: Math.floor(Date.now() / 1000) };
    await store.setJSON(`fb/${sid}/${keyHash}`, row);
    await store.set(ipKey, String(used + 1));
    return json({ saved: true });
  }

  if (req.method === "GET") {
    const token = process.env.ADMIN_TOKEN;
    if (!token || req.headers.get("x-admin-token") !== token) return json({ detail: "forbidden" }, 403);
    const { blobs } = await store.list();
    // Exports both the new rows (fb/<sid>/<keyHash>) and rows saved before sessions existed.
    const rows = await Promise.all(blobs.filter(b => !b.key.startsWith("rate/")).map(b => store.get(b.key, { type: "json" })));
    const lines = rows.filter(r => r && r.traits).map(r => JSON.stringify(r)).join("\n");
    return new Response(lines + "\n", { headers: { "content-type": "application/x-ndjson" } });
  }

  return json({ detail: "method not allowed" }, 405);
};

export const config = { path: "/api/feedback" };
