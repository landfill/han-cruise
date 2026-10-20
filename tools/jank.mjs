// Main-thread jank on a phone-like setup, without a GPU: the real app (voyage running, HUD on)
// over the stub WebGPU in tools/fakegpu.js, in a phone viewport (412x915 CSS px, DPR 2.625,
// touch, ?mobile=1) with the CPU slowed by CDP throttling. Records the rAF interval
// distribution, hitches, long tasks, GC pauses and the main-thread time by activity from a
// Chrome trace, plus canvas resizes and WebGPU calls per frame.
// GPU time is not part of this measurement (the stub does no GPU work).
// Usage: node tools/jank.mjs [--url http://127.0.0.1:4173/] [--params ...] [--cpu 4] [--seconds 60]
//          [--warm 5] [--drag] [--tod golden|day|night] [--noblur] [--out captures/perf/jank.json] [--label name] [--chrome path]
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { findChrome, FAKEGPU_SCRIPT } from './browser.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const URL0 = opt('url', 'http://127.0.0.1:4173/');
const params = opt('params', '');
const cpu = Number(opt('cpu', 4));
const seconds = Number(opt('seconds', 60));
const warm = Number(opt('warm', 5));
const drag = args.includes('--drag');
const tod = opt('tod', 'golden');
const noblur = args.includes('--noblur');
const OUTF = opt('out', 'captures/perf/jank.json');
const label = opt('label', URL0);
const exe = opt('chrome', process.env.CHROME || findChrome());

const browser = await chromium.launch({ executablePath: exe, headless: true });
const ctx = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2.625, isMobile: true, hasTouch: true });
await ctx.addInitScript({ path: FAKEGPU_SCRIPT });
// --noblur: the same page without backdrop-filter (to measure what the blurred panels cost)
if (noblur) await ctx.addInitScript(() => { addEventListener('DOMContentLoaded', () => { const st = document.createElement('style'); st.textContent = '*{backdrop-filter:none!important;-webkit-backdrop-filter:none!important}'; document.head.append(st); }); });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const cdp = await ctx.newCDPSession(page);

await page.goto(`${URL0}${URL0.includes('?') ? '&' : '?'}autostart&mute&tod=${tod}&mobile=1${params ? '&' + params : ''}`);
await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
await cdp.send('Emulation.setCPUThrottlingRate', { rate: cpu });
await page.waitForTimeout(warm * 1000);

// page-side recorder: rAF intervals, canvas size changes, long tasks, WebGPU call count
await page.evaluate(() => {
  const r = window.__rec = { iv: [], sizes: 0, longTasks: [], calls0: window.__gpuCalls.n, frames0: window.__app.frames() };
  const c = document.getElementById('c');
  let w = c.width, h = c.height, last = performance.now();
  r.stop = false;
  const f = (now) => {
    r.iv.push(now - last); last = now;
    if (c.width !== w || c.height !== h) { r.sizes++; w = c.width; h = c.height; }
    if (!r.stop) requestAnimationFrame(f);
  };
  requestAnimationFrame(f);
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) r.longTasks.push(e.duration); }).observe({ type: 'longtask' }); } catch (e) { /* not supported */ }
});

const traceChunks = [];
cdp.on('Tracing.dataCollected', (e) => traceChunks.push(...e.value));
const traceDone = new Promise((res) => cdp.once('Tracing.tracingComplete', res));
await cdp.send('Tracing.start', { categories: 'devtools.timeline,v8,disabled-by-default-v8.gc,blink,cc,viz', transferMode: 'ReportEvents' });

// optional: look around now and then like a phone user (touch drags)
const t0 = Date.now();
while (Date.now() - t0 < seconds * 1000) {
  if (drag) {
    const y = 450, x0 = 120 + Math.random() * 60;
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: x0, y }] });
    for (let i = 1; i <= 20; i++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x0 + i * 8, y: y + i * 2 }] }); await page.waitForTimeout(16); }
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  }
  await page.waitForTimeout(2000);
}

const rec = await page.evaluate(() => { const r = window.__rec; r.stop = true; return { iv: r.iv.slice(3), sizes: r.sizes, longTasks: r.longTasks, calls: window.__gpuCalls.n - r.calls0, frames: window.__app.frames() - r.frames0, stats: (({ W, H, dynScale }) => ({ W, H, dynScale }))(window.__app.stats()) }; });
await cdp.send('Tracing.end');
await traceDone;
await browser.close();

// ---- trace: main-thread time by activity, GC pauses ----
const main = new Set();
const threadName = new Map();
for (const e of traceChunks) if (e.name === 'thread_name') { threadName.set(`${e.pid}:${e.tid}`, e.args?.name); if (e.args?.name === 'CrRendererMain') main.add(`${e.pid}:${e.tid}`); }
// busy time of the compositor and GPU-side threads (top-level trace slices), ms per second
const busy = {};
{
  const byThread = new Map();
  for (const e of traceChunks) {
    if (e.ph !== 'X') continue;
    const n = threadName.get(`${e.pid}:${e.tid}`);
    if (!/^(VizCompositorThread|CrGpuMain|Compositor|CrRendererMain)$/.test(n || '')) continue;
    (byThread.get(n) || byThread.set(n, []).get(n)).push(e);
  }
  for (const [n, list] of byThread) {
    list.sort((a, b) => a.ts - b.ts);
    let until = -Infinity, sum = 0;
    for (const e of list) { if (e.ts >= until) { sum += (e.dur || 0) / 1000; until = e.ts + (e.dur || 0); } }
    busy[n] = sum;
  }
}
const ACT = {
  FunctionCall: 'script', 'v8.callFunction': 'script', TimerFire: 'script', FireAnimationFrame: 'rAF callback', EventDispatch: 'script',
  UpdateLayoutTree: 'style', Layout: 'layout', PrePaint: 'paint', Paint: 'paint', Layerize: 'paint', UpdateLayer: 'paint',
  'ProxyMain::BeginMainFrame::Commit': 'commit',
};
const byAct = {};
const gc = [];
const rafMs = [];
// count only top-level activities (a FunctionCall inside a rAF callback is part of that callback)
const evs = traceChunks.filter((e) => e.ph === 'X' && main.has(`${e.pid}:${e.tid}`)).sort((a, b) => a.ts - b.ts);
let coveredUntil = -Infinity;
for (const e of evs) {
  const ms = (e.dur || 0) / 1000;
  if (e.name === 'MinorGC' || e.name === 'MajorGC') { gc.push({ kind: e.name, ms }); continue; }
  if (e.name === 'FireAnimationFrame') rafMs.push(ms);
  const a = ACT[e.name];
  if (!a || e.ts < coveredUntil) continue;
  byAct[a] = (byAct[a] || 0) + ms;
  coveredUntil = e.ts + (e.dur || 0);
}
const rs = rafMs.slice().sort((a, b) => a - b);
const rpct = (p) => rs[Math.min(rs.length - 1, Math.floor(rs.length * p))] || 0;

const iv = rec.iv;
const sorted = iv.slice().sort((a, b) => a - b);
const pct = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] || 0;
const dur = iv.reduce((a, b) => a + b, 0) / 1000;
const out = {
  label, cpu, tod, seconds, drag, noblur, params, frames: iv.length,
  fps: +(iv.length / dur).toFixed(1),
  p50: +pct(0.5).toFixed(1), p95: +pct(0.95).toFixed(1), p99: +pct(0.99).toFixed(1), max: +sorted.at(-1).toFixed(1),
  over25pct: +(100 * iv.filter((x) => x > 25).length / iv.length).toFixed(2),
  hitches50: iv.filter((x) => x > 50).length,
  hitches100: iv.filter((x) => x > 100).length,
  longTasks: rec.longTasks.length, longTaskMs: +rec.longTasks.reduce((a, b) => a + b, 0).toFixed(0),
  rafCallbackMs: { p50: +rpct(0.5).toFixed(2), p95: +rpct(0.95).toFixed(2), p99: +rpct(0.99).toFixed(2), max: +(rs.at(-1) || 0).toFixed(1) },
  canvasResizes: rec.sizes,
  gpuCallsPerFrame: +(rec.calls / Math.max(1, rec.frames)).toFixed(0),
  gc: { minor: gc.filter((g) => g.kind === 'MinorGC').length, major: gc.filter((g) => g.kind === 'MajorGC').length, totalMs: +gc.reduce((a, g) => a + g.ms, 0).toFixed(1), maxMs: +Math.max(0, ...gc.map((g) => g.ms)).toFixed(1) },
  threadBusyMsPerSec: Object.fromEntries(Object.entries(busy).map(([k, v]) => [k, +(v / dur).toFixed(1)])),
  mainThreadMsPerSec: Object.fromEntries(Object.entries(byAct).map(([k, v]) => [k, +(v / dur).toFixed(1)])),
  end: rec.stats, errors: errors.slice(0, 10),
};
console.log(JSON.stringify(out, null, 1));
fs.mkdirSync(path.dirname(OUTF), { recursive: true });
let all = [];
try { all = JSON.parse(fs.readFileSync(OUTF, 'utf8')); } catch (e) { /* new file */ }
all.push(out);
fs.writeFileSync(OUTF, JSON.stringify(all, null, 1));
