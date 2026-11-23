// tests/smoke.mjs — open the real site in Chromium, take the quiz, refine, search, and fail on any error.
// The CPU is slowed 4x to approximate a mid-range phone; the timings it prints are the performance profile.
// Usage:  (cd tests && npm install && npx playwright install chromium)
//         python -m http.server 8080 -d site      (in another terminal)
//         BASE_URL=http://localhost:8080 node tests/smoke.mjs        [BROWSER_CHANNEL=msedge to use installed Edge]
import { chromium } from "playwright";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.BASE_URL || "http://localhost:8080";
const CPU_SLOWDOWN = Number(process.env.CPU_SLOWDOWN || 4);
const LIMITS = { modelLoadMs: 20000, recommendMs: 2000 };        // under CPU_SLOWDOWN
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// Serve with the same Content-Security-Policy as Netlify (python's server doesn't read _headers).
const CSP = readFileSync(join(root, "site", "_headers"), "utf8").match(/Content-Security-Policy:\s*(.+)/)[1].trim();

const problems = [];
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || undefined });
const page = await browser.newPage();
await page.route("**/*", async route => {
  const res = await route.fetch();
  const type = route.request().resourceType();
  const headers = { ...res.headers() };
  const url = route.request().url();
  if (type === "document" || url.endsWith("worker.js")) headers["content-security-policy"] = CSP;
  // python's test server takes MIME types from the OS; set the two the engine needs (Netlify does this itself)
  if (url.endsWith(".mjs")) headers["content-type"] = "text/javascript";
  if (url.endsWith(".wasm")) headers["content-type"] = "application/wasm";
  await route.fulfill({ response: res, headers });
});
page.on("pageerror", e => problems.push(`page error: ${e.message}`));
page.on("console", m => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
await page.addInitScript(() => addEventListener("securitypolicyviolation", e =>
  console.error(`CSP violation: ${e.violatedDirective} ${e.blockedURI}`)));
const cdp = await page.context().newCDPSession(page);
await cdp.send("Emulation.setCPUThrottlingRate", { rate: CPU_SLOWDOWN });

const t0 = Date.now();
await page.goto(BASE + "/index.html");
await page.waitForFunction(() => document.querySelector(".status")?.textContent.includes("Model ready"), null, { timeout: 60000 });
const modelLoadMs = Date.now() - t0;

const lastCards = () => page.locator(".cards").last().locator(".card");
const answer = async label => {
  await page.locator(".quick button:not([disabled])", { hasText: label }).last().click();
  await page.waitForFunction(() => !document.querySelector(".typing"), null, { timeout: 30000 });
};
const say = async text => {
  const before = await page.locator(".cards").count();
  await page.fill("#input", text);
  await page.press("#input", "Enter");
  await page.waitForFunction(n => document.querySelectorAll(".cards").length > n && !document.querySelector(".typing"), before, { timeout: 30000 });
};

for (const label of ["Men's", "Out with friends", "Fun and spontaneous", "Work and the office", "Spring", "Warm and spicy"]) await answer(label);
await page.waitForSelector(".cards .card");
const quizPicks = await lastCards().locator(".card-name").allInnerTexts();
if (quizPicks.length !== 3) problems.push(`expected 3 picks after the quiz, got ${quizPicks.length}`);
const badges = await lastCards().locator(".badge").allInnerTexts();
if (badges.some(b => b.includes("Women's"))) problems.push(`a women's fragrance was recommended to a men's profile: ${badges}`);

await say("fresher and spicier");
const refined = await lastCards().locator(".card-name").allInnerTexts();
await say("i love dior sauvage");
const searchText = await page.locator(".row.ai .txt").last().innerText();
if (!/Found it — Sauvage/.test(searchText)) problems.push(`search did not find Sauvage: ${searchText.slice(0, 120)}`);
const similar = await lastCards().locator(".card-name").allInnerTexts();
if (similar.length !== 3 || similar.some(n => /^Sauvage/.test(n))) problems.push(`similar-to picks look wrong: ${similar}`);

const stats = await page.evaluate(() => ({ n: STATS.recCount, avgMs: Math.round(STATS.recMs / Math.max(1, STATS.recCount)) }));
await browser.close();

const report = { cpuSlowdown: CPU_SLOWDOWN, modelLoadMs, recommendations: stats.n, avgRecommendMs: stats.avgMs, quizPicks, refined, similarToSauvage: similar };
console.log(JSON.stringify(report, null, 1));
if (modelLoadMs > LIMITS.modelLoadMs) problems.push(`model load ${modelLoadMs} ms > ${LIMITS.modelLoadMs} ms`);
if (stats.avgMs > LIMITS.recommendMs) problems.push(`recommendation ${stats.avgMs} ms > ${LIMITS.recommendMs} ms`);
if (problems.length) { console.error("\nSMOKE TEST FAILED:\n  " + problems.join("\n  ")); process.exit(1); }
console.log("\nsmoke test passed");
