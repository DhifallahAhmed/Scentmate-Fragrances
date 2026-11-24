// Netlify Function: anonymous health events from the page, so you know if the model fails for real visitors.
//   POST /api/log   (sendBeacon, text/plain JSON)  {type: "load"|"error"|"perf"|"share", ok?, ms?, workerMs?, msg?, src?, n?, avgMs?, ua}
//   GET  /api/log?day=YYYY-MM-DD   events of that day as JSONL + a summary line  (header X-Admin-Token = ADMIN_TOKEN)
// Never receives quiz answers. Stored in Netlify Blobs under log/<day>/..., capped per IP per day.
import { getStore } from "@netlify/blobs";

const TYPES = ["load", "error", "perf", "share"];   // "share" = someone shared their results (a count only)
const MAX_PER_IP_PER_DAY = 50;

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("").slice(0, 24);
}
const num = v => (Number.isFinite(Number(v)) ? Math.round(Number(v)) : undefined);
const str = (v, n) => (typeof v === "string" ? v.slice(0, n) : undefined);

export default async (req, context) => {
  const store = getStore("logs");

  if (req.method === "POST") {
    const raw = await req.text();
    if (raw.length > 2000) return new Response(null, { status: 413 });
    let e;
    try { e = JSON.parse(raw); } catch { return new Response(null, { status: 400 }); }
    if (!TYPES.includes(e?.type)) return new Response(null, { status: 422 });

    const day = new Date().toISOString().slice(0, 10);
    const ipKey = `rate/${day}/${await sha256((process.env.IP_SALT || "scentmate") + (context.ip || "unknown"))}`;
    const used = Number(await store.get(ipKey)) || 0;
    if (used >= MAX_PER_IP_PER_DAY) return new Response(null, { status: 204 });   // silently drop floods

    const event = {   // whitelist fields only
      type: e.type, ok: typeof e.ok === "boolean" ? e.ok : undefined, ms: num(e.ms), workerMs: num(e.workerMs),
      n: num(e.n), avgMs: num(e.avgMs), msg: str(e.msg, 300), src: str(e.src, 200), ua: str(e.ua, 160),
      ts: Math.floor(Date.now() / 1000),
    };
    await store.setJSON(`log/${day}/${event.ts}-${crypto.randomUUID()}`, event);
    await store.set(ipKey, String(used + 1));
    return new Response(null, { status: 204 });
  }

  if (req.method === "GET") {
    const token = process.env.ADMIN_TOKEN;
    if (!token || req.headers.get("x-admin-token") !== token) return new Response("forbidden", { status: 403 });
    const day = new URL(req.url).searchParams.get("day") || new Date().toISOString().slice(0, 10);
    const { blobs } = await store.list({ prefix: `log/${day}/` });
    const events = (await Promise.all(blobs.map(b => store.get(b.key, { type: "json" })))).filter(Boolean);
    const loads = events.filter(x => x.type === "load");
    const failed = loads.filter(x => x.ok === false).length;
    const times = loads.filter(x => x.ok).map(x => x.ms).sort((a, b) => a - b);
    const summary = { day, loads: loads.length, failed, failRate: loads.length ? +(failed / loads.length).toFixed(3) : 0,
                      loadMsMedian: times[Math.floor(times.length / 2)] ?? null, errors: events.filter(x => x.type === "error").length,
                      shares: events.filter(x => x.type === "share").length };
    const body = [JSON.stringify({ summary }), ...events.map(x => JSON.stringify(x))].join("\n") + "\n";
    return new Response(body, { headers: { "content-type": "application/x-ndjson" } });
  }

  return new Response(null, { status: 405 });
};

export const config = { path: "/api/log" };
