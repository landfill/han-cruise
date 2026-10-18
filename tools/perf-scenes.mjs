// Per-scene throughput (vsync off) at a fixed resolution, plus a sustained run that logs how frame
// time drifts over minutes (thermal behaviour). Results -> captures/perf.json
// Usage: node tools/perf-scenes.mjs [--url http://127.0.0.1:4173/] [--w 1920 --h 1080] [--sustain 180]
import fs from 'node:fs';
import { chromium } from 'playwright-core';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const URL0 = opt('url', 'http://127.0.0.1:4173/'), W = opt('w', 1920), H = opt('h', 1080);
const sustain = Number(opt('sustain', 180));
const scenes = (opt('scenes', 'golden_city,golden_sun,bridge_under,rail_approach,stern_wake,night_yeouido,night_rail,day_overview')).split(',');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const out = { resolution: `${W}x${H}`, scenes: {}, sustained: [] };
async function sample(seconds) {
  const ms = await page.evaluate(async (sec) => {
    const t = []; let last = performance.now();
    const end = last + sec * 1000;
    await new Promise((res) => { const f = (now) => { t.push(now - last); last = now; if (now < end) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
    return t.slice(5);
  }, seconds);
  ms.sort((a, b) => a - b);
  const mean = ms.reduce((a, b) => a + b, 0) / ms.length;
  return { mean: +mean.toFixed(2), p50: +ms[Math.floor(ms.length * 0.5)].toFixed(2), p95: +ms[Math.floor(ms.length * 0.95)].toFixed(2), max: +ms[ms.length - 1].toFixed(2) };
}
for (const sc of scenes) {
  await page.goto(`${URL0}?scene=${sc}&w=${W}&h=${H}&ui=0&mute&fixeddt`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  await page.waitForFunction(() => window.__app.frames() > 120, null, { timeout: 120000 });
  const r = await sample(4);
  const gpu = (await page.evaluate(() => window.__app.stats())).gpuMs;
  out.scenes[sc] = { ...r, gpuMs: +gpu.toFixed(2) };
  console.log(`${sc.padEnd(16)} frame mean ${r.mean} ms  p95 ${r.p95}  max ${r.max}  | GPU ${gpu.toFixed(2)} ms`);
}
if (sustain) {
  await page.goto(`${URL0}?autostart&mute&tod=golden&w=${W}&h=${H}&s=600`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  for (let t = 0; t < sustain; t += 15) {
    const r = await sample(5);
    const gpu = (await page.evaluate(() => window.__app.stats())).gpuMs;
    out.sustained.push({ t, ...r, gpuMs: +gpu.toFixed(2) });
    console.log(`sustained t=${String(t).padStart(3)}s  frame mean ${r.mean} ms  p95 ${r.p95}  GPU ${gpu.toFixed(2)} ms`);
    await page.waitForTimeout(10000);
  }
}
fs.mkdirSync('captures', { recursive: true });
fs.writeFileSync('captures/perf.json', JSON.stringify(out, null, 2));
await browser.close();
