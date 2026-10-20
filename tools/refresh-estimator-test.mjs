// Scenarios for the screen refresh estimator in src/sim/dynres.js (createRefreshEstimator).
// rAF callbacks come on refresh boundaries, so every interval is a whole number of refreshes; the
// browser skips callbacks when the GPU is behind. Usage: node tools/refresh-estimator-test.mjs
import { createRefreshEstimator } from '../src/sim/dynres.js';

let seed = 1;
const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
// feed `seconds` of callbacks: each interval is k refreshes, k drawn from `mix` ({k: weight})
function feed(est, { hz, seconds, mix = { 1: 1 }, gpu, t0 = 0 }) {
  const R = 1000 / hz, ks = Object.entries(mix).flatMap(([k, w]) => Array(w).fill(Number(k)));
  let t = t0;
  while (t < t0 + seconds) {
    const d = ks[Math.floor(rnd() * ks.length)] * R + (rnd() - 0.5) * 0.4;   // +-0.2 ms timestamp jitter
    t += d / 1000;
    est.add(d, t, gpu);
  }
  return t;
}
const cases = [
  // [name, setup, expected Hz]
  ['120 Hz idle (loading)', (e) => feed(e, { hz: 120, seconds: 2, gpu: 0 }), 120],
  ['60 Hz, GPU far behind: every callback skipped once (all 33 ms)', (e) => { const t = feed(e, { hz: 60, seconds: 2, gpu: 0 }); feed(e, { hz: 60, seconds: 120, mix: { 2: 1 }, gpu: 31, t0: t }); }, 60],
  ['60 Hz, GPU behind: 16.7/33 ms mix', (e) => { const t = feed(e, { hz: 60, seconds: 2, gpu: 0 }); feed(e, { hz: 60, seconds: 120, mix: { 1: 1, 2: 3 }, gpu: 22, t0: t }); }, 60],
  ['120 Hz, GPU behind at a 60 fps pace: 25/33 ms mix', (e) => { const t = feed(e, { hz: 120, seconds: 2, gpu: 0 }); feed(e, { hz: 120, seconds: 120, mix: { 3: 1, 4: 1 }, gpu: 25, t0: t }); }, 120],
  ['120 -> 60 Hz (power saving), GPU 25 ms: 16.7/33 ms mix', (e) => { const t = feed(e, { hz: 120, seconds: 2, gpu: 0 }); feed(e, { hz: 60, seconds: 120, mix: { 1: 1, 2: 1 }, gpu: 25, t0: t }); }, 60],
  ['120 -> 60 Hz, GPU 10 ms (keeps up)', (e) => { const t = feed(e, { hz: 120, seconds: 2, gpu: 0 }); feed(e, { hz: 60, seconds: 120, gpu: 10, t0: t }); }, 60],
  ['60 -> 30 Hz screen, GPU 10 ms', (e) => { const t = feed(e, { hz: 60, seconds: 2, gpu: 0 }); feed(e, { hz: 30, seconds: 120, gpu: 10, t0: t }); }, 30],
  ['90 Hz idle, then GPU behind at a 45 fps pace: 22/33 ms mix', (e) => { const t = feed(e, { hz: 90, seconds: 2, gpu: 0 }); feed(e, { hz: 90, seconds: 120, mix: { 2: 1, 3: 1 }, gpu: 24, t0: t }); }, 90],
  ['144 Hz idle', (e) => feed(e, { hz: 144, seconds: 2, gpu: 0 }), 144],
  ['60 -> 120 Hz', (e) => { const t = feed(e, { hz: 60, seconds: 5, gpu: 8 }); feed(e, { hz: 120, seconds: 5, gpu: 8, t0: t }); }, 120],
];
let fails = 0;
for (const [name, setup, hz] of cases) {
  const e = createRefreshEstimator();
  setup(e);
  const got = 1000 / e.ms, ok = Math.abs(got - hz) < 1;
  if (!ok) fails++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${got.toFixed(1)} Hz (expected ${hz})`);
}
console.log(`${cases.length - fails}/${cases.length} passed`);
process.exit(fails ? 1 : 0);
