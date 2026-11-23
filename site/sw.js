/* sw.js — offline support + instant repeat visits.
   * model/files/* and vendor/<version>/* have content-hashed / versioned names -> cache-first, kept forever
     (old versions are pruned when a new config.json points elsewhere).
   * The page, worker.js and model/config.json -> network-first, falling back to the cache when offline.
   * /api/* is never cached. */
const CACHE = "scentmate-v1";
const SHELL = ["./", "index.html", "worker.js", "favicon.svg", "logo.svg", "model/config.json"];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {}).then(() => self.skipWaiting()));
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

const immutable = url => url.pathname.includes("/model/files/") || url.pathname.includes("/vendor/");

self.addEventListener("fetch", event => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;

  if (immutable(url)) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE);
      const hit = await cache.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })());
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const res = await fetch(req);
      if (res.ok) {
        cache.put(req, res.clone());
        if (url.pathname.endsWith("/model/config.json")) pruneOldModels(res.clone());
      }
      return res;
    } catch (err) {
      const hit = await cache.match(req);
      if (hit) return hit;
      throw err;
    }
  })());
});

/* Remove cached model files that the current config.json no longer references. */
async function pruneOldModels(res) {
  try {
    const cfg = await res.json();
    const keep = new Set(Object.values(cfg.files).map(f => "/model/" + f));
    const cache = await caches.open(CACHE);
    for (const req of await cache.keys()) {
      const p = new URL(req.url).pathname;
      if (p.includes("/model/files/") && ![...keep].some(k => p.endsWith(k))) await cache.delete(req);
    }
  } catch { /* best effort */ }
}
