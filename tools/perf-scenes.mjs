// Per-scene throughput (vsync off) at a fixed resolution, plus a sustained run that logs how frame
// time drifts over minutes (thermal behaviour). Results -> captures/perf.json
// Usage: node tools/perf-scenes.mjs [--url http://127.0.0.1:4173/] [--w 1920 --h 1080] [--sustain 180]
//        [--scenes a,b | all] [--params mobile=1] [--repeat 3] [--out captures/perf.json]
// With --repeat N each scene is measured N times (interleaved rounds, to spread thermal drift) and
// the median GPU time is reported.
import fs from 'node:fs';
import { chromium } from 'playwright-core';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const URL0 = opt('url', 'http://127.0.0.1:4173/'), W = opt('w', 1920), H = opt('h', 1080);
const sustain = Number(opt('sustain', 180));
const DEFAULT = 'golden_city,golden_sun,bridge_under,rail_approach,stern_wake,night_yeouido,night_rail,day_overview';
let scenes = opt('scenes', DEFAULT);
if (scenes === 'all') scenes = Object.keys((await import('../src/sim/scenes.js')).SCENES).join(',');
scenes = scenes.split(',');
const params = opt('params', '');
const repeat = Number(opt('repeat', 1));
const OUTF = opt('out', 'captures/perf.json');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const out = { resolution: `${W}x${H}`, params, repeat, scenes: {}, sustained: [] };
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
const runs = {};
for (let k = 0; k < repeat; k++) for (const sc of scenes) {
  await page.goto(`${URL0}?scene=${sc}&w=${W}&h=${H}&ui=0&mute&fixeddt${params ? '&' + params : ''}`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  await page.waitForFunction(() => window.__app.frames() > 120, null, { timeout: 120000 });
  const r = await sample(4);
  const st = await page.evaluate(() => window.__app.stats());
  (runs[sc] ||= []).push({ ...r, gpuMs: st.gpuMs, errors: st.errors.length });
  console.log(`[${k + 1}/${repeat}] ${sc.padEnd(16)} frame mean ${r.mean} ms  p95 ${r.p95}  max ${r.max}  | GPU ${st.gpuMs.toFixed(2)} ms  ${st.W}x${st.H}${st.errors.length ? '  ERR ' + st.errors.length : ''}`);
}
const med = (a) => { const b = a.slice().sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
for (const sc of scenes) {
  const rs = runs[sc];
  out.scenes[sc] = { mean: med(rs.map((r) => r.mean)), p95: med(rs.map((r) => r.p95)), gpuMs: +med(rs.map((r) => r.gpuMs)).toFixed(2), gpuRuns: rs.map((r) => +r.gpuMs.toFixed(2)), errors: Math.max(...rs.map((r) => r.errors)) };
}
console.log('median GPU ms: ' + scenes.map((sc) => `${sc} ${out.scenes[sc].gpuMs}`).join(', '));
if (sustain) {
  await page.goto(`${URL0}?autostart&mute&tod=golden&w=${W}&h=${H}&s=600${params ? '&' + params : ''}`);
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
fs.writeFileSync(OUTF, JSON.stringify(out, null, 2));
await browser.close();
