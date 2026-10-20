// Checks that two builds send the GPU exactly the same work: each build steps the same fixed
// scenes with a fixed time step over the stub WebGPU (tools/fakegpu.js ?gpulog), which hashes
// every WebGPU call with its arguments and uploaded bytes. Equal hashes = same buffers, same
// draws in the same order, same uniforms, i.e. the same images and the same GPU time.
// Usage: node tools/same-gpu-work.mjs --a http://127.0.0.1:4174/ --b http://127.0.0.1:4173/
//          [--scenes golden_city,night_yeouido] [--frames 120] [--params mobile=0&w=1920&h=1080]
import { chromium } from 'playwright-core';
import { findChrome, FAKEGPU_SCRIPT } from './browser.mjs';
import { SCENES } from '../src/sim/scenes.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const A = opt('a', 'http://127.0.0.1:4174/'), B = opt('b', 'http://127.0.0.1:4173/');
const FRAMES = Number(opt('frames', 120));
const params = opt('params', 'mobile=0&w=1920&h=1080');
let scenes = opt('scenes', 'all');
scenes = scenes === 'all' ? Object.keys(SCENES) : scenes.split(',');

const browser = await chromium.launch({ executablePath: findChrome(), headless: true });
async function run(base, sc) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  await ctx.addInitScript({ path: FAKEGPU_SCRIPT });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${base}${base.includes("?") ? "&" : "?"}scene=${sc}&manual&fixeddt&gpulog&ui=0&mute&${params}`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  const h0 = await page.evaluate(() => window.__gpuHash);
  const h = await page.evaluate((n) => { for (let i = 0; i < n; i++) window.__app.step(1 / 60); return window.__gpuHash; }, FRAMES);
  await ctx.close();
  return { init: h0, frames: h, errors };
}
let same = 0;
for (const sc of scenes) {
  const a = await run(A, sc), b = await run(B, sc);
  const ok = a.frames === b.frames;
  if (ok) same++;
  console.log(`${sc.padEnd(22)} load ${a.init === b.init ? 'same' : 'DIFF'}  ${FRAMES} frames ${ok ? 'same' : 'DIFF'}  (${a.frames.toString(16)} / ${b.frames.toString(16)})${a.errors.length + b.errors.length ? '  errors: ' + [...a.errors, ...b.errors].slice(0, 2).join(' | ') : ''}`);
}
console.log(`${same}/${scenes.length} scenes: identical GPU work`);
await browser.close();
process.exit(same === scenes.length ? 0 : 1);
