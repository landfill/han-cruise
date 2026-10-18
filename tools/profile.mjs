// Frame-time profiler. Chrome runs with vsync and the frame-rate limit disabled, so the measured
// frame time is the real throughput of the GPU+CPU pipeline. Each pass group is disabled in turn
// (base runs are interleaved to expose thermal drift).
// Usage: node tools/profile.mjs [--scene golden_city] [--w 1920 --h 1080] [--groups refl,shadow,...]
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const scene = opt('scene', 'golden_city'), W = opt('w', 1920), H = opt('h', 1080);
const groups = opt('groups', 'refl,shadow,water,meshes,terrain,trees,sky').split(',');
const extra = opt('params', '');

const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

async function measure(skip) {
  await page.goto(`http://127.0.0.1:5173/?scene=${scene}&w=${W}&h=${H}&ui=0&mute&fixeddt${skip ? '&skip=' + skip : ''}${extra ? '&' + extra : ''}`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  await page.waitForFunction(() => window.__app.frames() > 200, null, { timeout: 120000 });
  const f0 = await page.evaluate(() => [window.__app.frames(), performance.now()]);
  await page.waitForTimeout(3000);
  const f1 = await page.evaluate(() => [window.__app.frames(), performance.now()]);
  const st = await page.evaluate(() => window.__app.stats());
  return { ms: (f1[1] - f0[1]) / (f1[0] - f0[0]), gpu: st.gpuMs };
}

const base = [];
const out = [];
for (const g of ['', ...groups, groups.join(',')]) {
  const b = await measure('');
  base.push(b.ms);
  const r = g ? await measure(g) : b;
  out.push([g || 'all passes', r, b]);
  console.log(`${(g || 'all passes').padEnd(46)} frame ${r.ms.toFixed(2)} ms (GPU ts ${r.gpu.toFixed(2)})${g ? `   base ${b.ms.toFixed(2)} -> pass ≈ ${(b.ms - r.ms).toFixed(2)} ms` : ''}`);
}
console.log(`base range ${Math.min(...base).toFixed(2)}–${Math.max(...base).toFixed(2)} ms`);
await browser.close();
