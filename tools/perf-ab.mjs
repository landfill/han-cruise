// Paired A/B throughput: the same fixed scenes rendered by two builds (e.g. the previous commit
// served from a worktree on another port, and the working tree), alternating A/B per scene inside
// each round and swapping the order every round, so thermal drift on a fanless machine hits both
// sides alike. Reports the median GPU time per scene and the paired change.
// Usage: node tools/perf-ab.mjs --a http://127.0.0.1:4174/ --b http://127.0.0.1:4173/
//          [--scenes all | a,b] [--w 1920 --h 1080] [--params mobile=1] [--rounds 3]
//          [--dyn] [--out captures/perf/ab.json]
// --dyn: no fixed render size; the page gets a W x H viewport at DPR 1 and ?dynq, so the pixel
// budget and the dynamic resolution apply, and the settled scale / render size are recorded.
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { SCENES } from '../src/sim/scenes.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const A = opt('a', 'http://127.0.0.1:4174/'), B = opt('b', 'http://127.0.0.1:4173/');
const W = Number(opt('w', 1920)), H = Number(opt('h', 1080));
const params = opt('params', '');
const rounds = Number(opt('rounds', 3));
const dynMode = args.includes('--dyn');
const OUTF = opt('out', 'captures/perf/ab.json');
let scenes = opt('scenes', 'all');
scenes = scenes === 'all' ? Object.keys(SCENES) : scenes.split(',');

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const page = await browser.newPage({ viewport: dynMode ? { width: W, height: H } : { width: 1280, height: 720 }, deviceScaleFactor: 1 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error' || /GPU error|validation/i.test(m.text())) errors.push(m.text()); });

async function measure(base, sc) {
  const size = dynMode ? 'dynq' : `w=${W}&h=${H}`;
  // a base URL may carry its own query (e.g. a feature toggle to A/B inside one build)
  await page.goto(`${base}${base.includes('?') ? '&' : '?'}scene=${sc}&${size}&ui=0&mute&fixeddt${params ? '&' + params : ''}`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  // dynamic mode: give the resolution controller time to settle (it steps every 1.5 s of sim time)
  await page.waitForFunction((n) => window.__app.frames() > n, dynMode ? 900 : 120, { timeout: 180000 });
  const ms = await page.evaluate(async () => {
    const t = []; let last = performance.now(); const end = last + 3000;
    await new Promise((res) => { const f = (now) => { t.push(now - last); last = now; if (now < end) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
    return t.slice(3);
  });
  const st = await page.evaluate(() => window.__app.stats());
  return { frame: ms.reduce((a, b) => a + b, 0) / ms.length, gpu: st.gpuMs, W: st.W, H: st.H, scale: st.dynScale ?? 1, errors: st.errors.length };
}

const runs = {};
for (let r = 0; r < rounds; r++) {
  for (const sc of scenes) {
    const order = r % 2 === 0 ? [['a', A], ['b', B]] : [['b', B], ['a', A]];
    for (const [k, url] of order) {
      const m = await measure(url, sc);
      ((runs[sc] ||= { a: [], b: [] })[k]).push(m);
    }
    const ra = runs[sc].a.at(-1), rb = runs[sc].b.at(-1);
    console.log(`[${r + 1}/${rounds}] ${sc.padEnd(16)} A ${ra.gpu.toFixed(2)} ms  B ${rb.gpu.toFixed(2)} ms  (${((rb.gpu / ra.gpu - 1) * 100).toFixed(1)}%)${dynMode ? `  scale ${ra.scale.toFixed(2)} -> ${rb.scale.toFixed(2)}  ${rb.W}x${rb.H}` : ''}`);
  }
}

const med = (a) => { const b = a.slice().sort((x, y) => x - y); const n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const out = { a: A, b: B, size: dynMode ? `viewport ${W}x${H} (dynamic)` : `${W}x${H}`, params, rounds, scenes: {} };
const lines = ['| 장면 | 이전 GPU (ms) | 이후 GPU (ms) | 변화 |' + (dynMode ? ' 이전 해상도 (배율) | 이후 해상도 (배율) |' : ''), '|---|---|---|---|' + (dynMode ? '---|---|' : '')];
for (const sc of scenes) {
  const ra = runs[sc].a, rb = runs[sc].b;
  // paired change: per-round ratio, then the median of those
  const ratios = ra.map((x, i) => rb[i].gpu / x.gpu);
  const o = {
    gpuA: +med(ra.map((x) => x.gpu)).toFixed(2), gpuB: +med(rb.map((x) => x.gpu)).toFixed(2),
    frameA: +med(ra.map((x) => x.frame)).toFixed(2), frameB: +med(rb.map((x) => x.frame)).toFixed(2),
    change: +((med(ratios) - 1) * 100).toFixed(1),
    runsA: ra.map((x) => +x.gpu.toFixed(2)), runsB: rb.map((x) => +x.gpu.toFixed(2)),
    scaleA: +med(ra.map((x) => x.scale)).toFixed(2), scaleB: +med(rb.map((x) => x.scale)).toFixed(2),
    resA: `${ra.at(-1).W}x${ra.at(-1).H}`, resB: `${rb.at(-1).W}x${rb.at(-1).H}`,
    errors: Math.max(...ra.map((x) => x.errors), ...rb.map((x) => x.errors)),
  };
  out.scenes[sc] = o;
  lines.push(`| ${sc} | ${o.gpuA.toFixed(2)} | ${o.gpuB.toFixed(2)} | ${o.change > 0 ? '+' : ''}${o.change.toFixed(1)}% |` + (dynMode ? ` ${o.resA} (${o.scaleA.toFixed(2)}) | ${o.resB} (${o.scaleB.toFixed(2)}) |` : ''));
}
out.consoleErrors = errors.slice(0, 20);
console.log('\n' + lines.join('\n'));
console.log(`console errors: ${errors.length}`);
fs.mkdirSync(path.dirname(OUTF), { recursive: true });
fs.writeFileSync(OUTF, JSON.stringify(out, null, 2));
fs.writeFileSync(OUTF.replace(/\.json$/, '.md'), lines.join('\n') + '\n');
await browser.close();
