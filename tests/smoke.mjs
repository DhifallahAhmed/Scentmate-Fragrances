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

for (const label of ["Men's", "Out with friends", "Fun and spontaneous", "Work and the office", "Spring", "Warm and spicy", "Up to $200"]) await answer(label);
await page.waitForSelector(".cards .card");
const quizPicks = await lastCards().locator(".card-name").allInnerTexts();
if (quizPicks.length !== 3) problems.push(`expected 3 picks after the quiz, got ${quizPicks.length}`);
const badges = await lastCards().locator(".badge").allInnerTexts();
if (badges.some(b => b.includes("Women's"))) problems.push(`a women's fragrance was recommended to a men's profile: ${badges}`);
const prices = await lastCards().locator(".price-tag b").allInnerTexts();
if (prices.some(p => p === "$$$")) problems.push(`budget "Up to $200" but got a $$$ pick: ${prices}`);
if (await lastCards().locator("svg.bottle").count() !== 3) problems.push("bottle art missing on cards");
const findHref = await lastCards().first().locator("a.find").getAttribute("href");
if (!findHref?.startsWith("https://www.google.com/search?")) problems.push(`"Where to buy" link looks wrong: ${findHref}`);
// Save the first pick; the shortlist counter and the stored list must follow.
await lastCards().first().locator("button.save").click();
if (await lastCards().first().locator("button.save").getAttribute("aria-pressed") !== "true") problems.push("save button did not toggle");
const savedCount = await page.evaluate(() => loadSaved().length);
if (savedCount !== 1) problems.push(`expected 1 saved fragrance, got ${savedCount}`);
if (await page.locator(".profile .share-btn").count() < 1) problems.push("share button missing on the results profile");
const shareLink = await page.evaluate(() => shareUrl());
if (!/#r=1330031$/.test(shareLink)) problems.push(`share link does not encode the answers: ${shareLink}`);

await say("fresher and spicier");
const refined = await lastCards().locator(".card-name").allInnerTexts();
await say("i love dior sauvage");
const searchText = await page.locator(".row.ai .txt").last().innerText();
if (!/Found it — Sauvage/.test(searchText)) problems.push(`search did not find Sauvage: ${searchText.slice(0, 120)}`);
const similar = await lastCards().locator(".card-name").allInnerTexts();
if (similar.length !== 3 || similar.some(n => /^Sauvage/.test(n))) problems.push(`similar-to picks look wrong: ${similar}`);

// The "Already love a fragrance?" picker: type, see suggestions, pick with the keyboard.
await page.fill("#fav-input", "");
await page.type("#fav-input", "coco mad", { delay: 40 });
await page.waitForSelector('#fav-list li[role="option"]:not(.fav-empty)', { timeout: 15000 });
const suggestions = await page.locator('#fav-list li[role="option"] .fav-name').allInnerTexts();
if (!suggestions[0]?.startsWith("Coco Mademoiselle")) problems.push(`picker suggestions look wrong: ${suggestions}`);
if (await page.getAttribute("#fav-input", "aria-expanded") !== "true") problems.push("picker: aria-expanded not set");
const beforePick = await page.locator(".cards").count();
await page.press("#fav-input", "Enter");
await page.waitForFunction(n => document.querySelectorAll(".cards").length > n && !document.querySelector(".typing"), beforePick, { timeout: 30000 });
const pickText = await page.locator(".row.ai .txt").last().innerText();
if (!/Found it — Coco Mademoiselle/.test(pickText)) problems.push(`picker did not anchor on Coco Mademoiselle: ${pickText.slice(0, 120)}`);
const pickBadges = await lastCards().locator(".badge").allInnerTexts();
if (pickBadges.some(b => b.includes("Men's"))) problems.push(`picked a women's favourite but got men's picks: ${pickBadges}`);
const similarToCoco = await lastCards().locator(".card-name").allInnerTexts();

const stats = await page.evaluate(() => ({ n: STATS.recCount, avgMs: Math.round(STATS.recMs / Math.max(1, STATS.recCount)) }));

// Open the shared link fresh: it must replay the quiz and show the same picks; the shortlist must survive the reload.
await page.goto("about:blank");
await page.goto(shareLink);
await page.waitForSelector(".cards .card", { timeout: 60000 });
const sharedPicks = await lastCards().locator(".card-name").allInnerTexts();
if (sharedPicks.join("|") !== quizPicks.join("|")) problems.push(`shared link shows different picks: ${sharedPicks} vs ${quizPicks}`);
if (page.url().includes("#r=")) problems.push("share hash not cleared after replay");
const shortlistLabel = await page.locator(".margin .saved-btn, .topbar .saved-btn").first().innerText();
if (!/\(1\)/.test(shortlistLabel)) problems.push(`shortlist count not kept after reload: ${shortlistLabel}`);
await page.locator(".saved-btn:visible").first().click();
await page.waitForSelector(".shortlist li", { timeout: 15000 });
await browser.close();

const report = { cpuSlowdown: CPU_SLOWDOWN, modelLoadMs, recommendations: stats.n, avgRecommendMs: stats.avgMs, quizPicks, quizPrices: prices, sharedPicks, refined, similarToSauvage: similar,
                 pickerSuggestions: suggestions, similarToCocoMademoiselle: similarToCoco };
console.log(JSON.stringify(report, null, 1));
if (modelLoadMs > LIMITS.modelLoadMs) problems.push(`model load ${modelLoadMs} ms > ${LIMITS.modelLoadMs} ms`);
if (stats.avgMs > LIMITS.recommendMs) problems.push(`recommendation ${stats.avgMs} ms > ${LIMITS.recommendMs} ms`);
if (problems.length) { console.error("\nSMOKE TEST FAILED:\n  " + problems.join("\n  ")); process.exit(1); }
console.log("\nsmoke test passed");
