// Offline simulation of the phone dynamic-resolution controller: the previous controller (copied
// from src/main.js before this change) against src/sim/dynres.js, on the same modelled GPU load.
// No browser, no GPU: it checks the control logic (how often the render size changes, the
// presented frame cadence) under assumptions stated below, not real phone timings.
//
// Model
// - GPU time per frame at render scale s: F * (FIXED + (base(t) - FIXED) * s^2), base(t) = a
//   3-minute voyage built from the measured 1.4 MP per-scene GPU times on the Adreno X1-85
//   (docs/VERIFICATION.md 9.3: open river 23-26 ms, under a bridge / close to the hull 11 ms,
//   lighter stretches ~15 ms), with smooth transitions, +1 ms on reflection frames (every 2nd),
//   +-6% per-frame noise. F = device speed relative to the Adreno X1-85 (unknown for real phones,
//   so several values are run).
// - rAF fires on vsync while fewer than 2 frames are queued on the GPU; the GPU runs frames in
//   order; a frame is shown at the first vsync after it finishes (a later frame finishing in the
//   same refresh replaces it).
// - A render-size change adds RESIZE_MS of GPU time to that frame (allocation and clearing of
//   the render targets; an assumption, the real cost was not measured on a phone).
// - GPU-timestamp path: smoothed GPU time sampled every 7th frame (as in the renderer); frame-time
//   path: rAF intervals.
// Usage: node tools/dynres-sim.mjs [--seconds 180] [--seeds 3] [--resize 25] [--out file]
import fs from 'node:fs';
import { createDynRes, paceDue, WINDOW } from '../src/sim/dynres.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const SECONDS = Number(opt('seconds', 180));
const SEEDS = Number(opt('seeds', 3));
const RESIZE_MS = Number(opt('resize', 25));
const OUTF = opt('out', '');
const FIXED = 3;   // ms that do not scale with the pixel count (FFT, sky LUTs, cloud dome, env cube)

// voyage: [duration s, 1.4 MP GPU ms on the Adreno X1-85]
const VOYAGE = [[20, 25], [15, 24], [6, 11], [20, 23.5], [25, 26], [12, 15], [20, 23], [6, 11], [25, 25], [15, 24.5], [16, 15]];
function baseAt(t) {
  t %= VOYAGE.reduce((a, v) => a + v[0], 0);
  let acc = 0;
  for (let i = 0; i < VOYAGE.length; i++) {
    const [d, ms] = VOYAGE[i];
    if (t < acc + d) {
      const prev = VOYAGE[(i + VOYAGE.length - 1) % VOYAGE.length][1];
      const k = Math.min(1, (t - acc) / 3);   // 3 s transition from the previous stretch
      return prev + (ms - prev) * (k * k * (3 - 2 * k));
    }
    acc += d;
  }
  return VOYAGE[0][1];
}

// ---- the previous controller, as it was in src/main.js ----
function legacyController(hasTS) {
  const dyn = { scale: 1, timer: 0, intervals: [], calm: 0, ceil: 1, ceilAge: 0 };
  return {
    get scale() { return dyn.scale; }, pace: 1,
    frame(dt, rawMs, gpuMs) {
      if (rawMs < 100) dyn.intervals.push(rawMs);
      dyn.timer += dt;
      if (dyn.timer < 1.5) return;
      dyn.timer = 0;
      const iv = dyn.intervals; dyn.intervals = [];
      if (!hasTS) {
        if (iv.length < 20) return;
        let missed = 0;
        for (const x of iv) if (x > 25) missed++;
        const r = missed / iv.length;
        dyn.ceilAge += 1.5;
        if (dyn.ceilAge > 30) dyn.ceil = 1;
        if (r > 0.08) { dyn.ceil = dyn.scale; dyn.ceilAge = 0; dyn.calm = 0; dyn.scale = Math.max(0.6, dyn.scale * (r > 0.3 ? 0.85 : 0.93)); }
        else if (r < 0.01 && ++dyn.calm >= 3) { dyn.calm = 0; dyn.scale = Math.max(dyn.scale, Math.min(1, dyn.ceil * 0.97, dyn.scale * 1.04)); }
        return;
      }
      const g = gpuMs;
      if (!g) return;
      if (g > 14.5) dyn.scale = Math.max(0.6, dyn.scale * Math.sqrt(13 / g));
      else if (g < 10.5 && dyn.scale < 1) dyn.scale = Math.min(1, dyn.scale * 1.06);
    },
  };
}

// ---- the new controller, driven the way src/main.js drives it ----
function newController(hasTS, V) {
  const c = createDynRes({ hasTimestamps: hasTS });
  let timer = 0, iv = [], t = 0, done = [];
  return {
    get scale() { return c.scale; }, get pace() { return c.pace; }, paced: true,
    done(ms) { if (!hasTS) done.push(ms); },
    frame(dt, rawMs, gpuMs) {
      t += dt;
      if (rawMs < 150) iv.push(rawMs);
      timer += dt;
      if (timer < WINDOW) return;
      timer = 0;
      c.update({ gpuMs, intervals: iv, doneMs: done, vsyncMs: V, now: t });
      iv = []; done = [];
    },
  };
}

function rng(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

function run({ ctrl, hz, F, hasTS, seed }) {
  const V = 1000 / hz, rnd = rng(seed);
  const end = SECONDS * 1000;
  let gpuFree = 0, frameNo = 0, lastRaf = 0, lastRender = -1e9, gpuEma = 0, sizeKey = null;
  const inflight = [];          // GPU end times of queued frames
  let shownAt = -1, shownIdx = -1;
  const shows = [];             // vsync times at which a new frame appeared
  let resizes = 0, scaleSum = 0, scaleN = 0, paced30 = 0;
  for (let k = 1; k * V < end; k++) {
    const now = k * V;
    // frames finishing by this vsync: the last one is shown
    let newest = -1;
    for (let i = inflight.length - 1; i >= 0; i--) if (inflight[i].end <= now) { newest = Math.max(newest, inflight[i].idx); inflight.splice(i, 1); }
    if (newest > shownIdx) { shownIdx = newest; shownAt = now; shows.push(now); }
    if (inflight.length >= 2) continue;           // rAF throttled: GPU is two frames behind
    // rAF callback
    if (ctrl.paced && !paceDue(now, lastRender, ctrl.pace, V)) continue;
    const rawMs = now - lastRender;
    const dt = Math.min(0.1, rawMs / 1000);
    lastRender = now; lastRaf = now;
    const s = ctrl.scale;
    const key = ctrl.paced ? s : Math.round(s * 1578);   // legacy: any scale change changes the render size
    let extra = 0;
    if (sizeKey !== null && key !== sizeKey) { resizes++; extra = RESIZE_MS; }
    sizeKey = key;
    const base = baseAt(now / 1000);
    let g = F * (FIXED + (base - FIXED) * s * s) + (frameNo % 2 === 0 ? F * 1.0 : 0);
    g *= 1 + (rnd() - 0.5) * 0.12;
    const start = Math.max(now + 2, gpuFree);   // ~2 ms of CPU before submit
    gpuFree = start + g + extra;
    inflight.push({ end: gpuFree, idx: frameNo });
    // submit -> onSubmittedWorkDone, with 0-2 ms until the promise callback runs
    if (ctrl.done) ctrl.done(gpuFree - (now + 2) + rnd() * 2);
    if (frameNo % 7 === 0) gpuEma = gpuEma ? gpuEma * 0.8 + g * 0.2 : g;
    frameNo++;
    if (now > 10000) { scaleSum += s * s; scaleN++; if (ctrl.pace === 2) paced30++; }
    ctrl.frame(dt, rawMs, gpuEma);
  }
  void lastRaf; void shownAt;
  // presented cadence after a 10 s warm-up
  const sh = shows.filter((x) => x > 10000);
  const iv = []; for (let i = 1; i < sh.length; i++) iv.push(sh[i] - sh[i - 1]);
  const sorted = iv.slice().sort((a, b) => a - b);
  let uneven = 0; for (let i = 1; i < iv.length; i++) if (Math.abs(iv[i] - iv[i - 1]) > V / 2) uneven++;
  const dur = (SECONDS - 10);
  return {
    fps: iv.length / dur,
    uneven: 100 * uneven / Math.max(1, iv.length - 1),
    over25: 100 * iv.filter((x) => x > 25).length / iv.length,
    hitch50: iv.filter((x) => x > 50).length,
    p95: sorted[Math.floor(sorted.length * 0.95)],
    resizesPerMin: resizes / (SECONDS / 60),
    pixels: scaleSum / scaleN,
    at30: 100 * paced30 / Math.max(1, scaleN),
  };
}

const rows = [];
for (const hasTS of [true, false]) for (const hz of [60, 90, 120, 144]) for (const F of [0.6, 1.0, 1.5, 2.0]) {
  const res = {};
  for (const [name, mk] of [['old', legacyController], ['new', newController]]) {
    const acc = {};
    for (let seed = 1; seed <= SEEDS; seed++) {
      const r = run({ ctrl: mk(hasTS, 1000 / hz), hz, F, hasTS, seed: seed * 7919 });
      for (const k in r) acc[k] = (acc[k] || 0) + r[k] / SEEDS;
    }
    res[name] = acc;
  }
  rows.push({ hasTS, hz, F, ...res });
}

const f1 = (x) => x.toFixed(1), f0 = (x) => x.toFixed(0);
console.log(`model: ${SECONDS} s voyage, ${SEEDS} seeds, resize cost ${RESIZE_MS} ms GPU (assumed)`);
console.log('| GPU 시간 측정 | 화면 | 기기 배율 F | 크기 변경/분 이전→이후 | 표시 fps | 불규칙 간격 % | 50 ms 초과 히치 | 화소 비율 | 30 fps 시간 % |');
console.log('|---|---|---|---|---|---|---|---|---|');
for (const r of rows) {
  const o = r.old, n = r.new;
  console.log(`| ${r.hasTS ? '타임스탬프' : '프레임 간격'} | ${r.hz} Hz | ${r.F} | ${f1(o.resizesPerMin)} → ${f1(n.resizesPerMin)} | ${f0(o.fps)} → ${f0(n.fps)} | ${f1(o.uneven)} → ${f1(n.uneven)} | ${f0(o.hitch50)} → ${f0(n.hitch50)} | ${o.pixels.toFixed(2)} → ${n.pixels.toFixed(2)} | ${f0(n.at30)} |`);
}
if (OUTF) fs.writeFileSync(OUTF, JSON.stringify({ seconds: SECONDS, seeds: SEEDS, resizeMs: RESIZE_MS, rows }, null, 1));
