// Netlify Function: stores "I'd wear this" clicks so the model can be retrained on real choices.
//   POST /api/feedback   {traits: {romantic: 1.2, ...}, gender: "women"|"men"|"unisex"|null, key_hash: 2166136261}
//   key_hash = stable 32-bit id of the fragrance (from site/model/items.json), unchanged across model rebuilds.
//   GET  /api/feedback   download everything as feedback.jsonl  (header X-Admin-Token = ADMIN_TOKEN)
// Data lives in Netlify Blobs (free, built in). Set ADMIN_TOKEN under
// Site configuration → Environment variables.
import { getStore } from "@netlify/blobs";

const TRAITS = ["adventurous", "romantic", "professional", "playful", "mysterious", "calm", "confident", "cozy"];
const GENDERS = ["women", "men", "unisex"];

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

export default async (req) => {
  const store = getStore("feedback");

  if (req.method === "POST") {
    let body;
    try { body = await req.json(); } catch { return json({ detail: "invalid JSON" }, 400); }

    const keyHash = Number(body?.key_hash);
    if (!Number.isInteger(keyHash) || keyHash < 0 || keyHash > 0xFFFFFFFF) return json({ detail: "invalid key_hash" }, 422);

    // Keep only known traits with sane values — never store arbitrary client data.
    const traits = {};
    for (const t of TRAITS) {
      const v = Number(body?.traits?.[t]);
      if (Number.isFinite(v) && v > 0) traits[t] = Math.min(v, 20);
    }
    const row = {
      traits,
      gender: GENDERS.includes(body?.gender) ? body.gender : null,
      key_hash: keyHash,
      ts: Math.floor(Date.now() / 1000),
    };
    await store.setJSON(`${row.ts}-${crypto.randomUUID()}`, row);
    return json({ saved: true });
  }

  if (req.method === "GET") {
    const token = process.env.ADMIN_TOKEN;
    if (!token || req.headers.get("x-admin-token") !== token) return json({ detail: "forbidden" }, 403);
    const { blobs } = await store.list();
    const rows = await Promise.all(blobs.map(b => store.get(b.key, { type: "json" })));
    const lines = rows.filter(Boolean).map(r => JSON.stringify(r)).join("\n");
    return new Response(lines + "\n", { headers: { "content-type": "application/x-ndjson" } });
  }

  return json({ detail: "method not allowed" }, 405);
};

export const config = { path: "/api/feedback" };
