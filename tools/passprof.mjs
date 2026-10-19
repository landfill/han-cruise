// Per-pass GPU time (timestamp pairs, ?passprof) for fixed scenes.
// Usage: node tools/passprof.mjs [--url http://127.0.0.1:4173/] [--scenes a,b] [--w 1578 --h 887] [--params mobile=1]
import { chromium } from 'playwright-core';
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const URL0 = opt('url', 'http://127.0.0.1:4173/'), W = opt('w', 1920), H = opt('h', 1080), params = opt('params', '');
const scenes = opt('scenes', 'golden_city,stern_wake,night_nodeul').split(',');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--disable-gpu-vsync', '--disable-frame-rate-limit'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
for (const sc of scenes) {
  await page.goto(`${URL0}?scene=${sc}&w=${W}&h=${H}&ui=0&mute&fixeddt&passprof${params ? '&' + params : ''}`);
  await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  await page.waitForFunction(() => window.__app.frames() > 400, null, { timeout: 180000 });
  const st = await page.evaluate(() => window.__app.stats());
  const p = st.passMs || {};
  const sum = Object.values(p).reduce((a, b) => a + b, 0);
  console.log(`${sc.padEnd(16)} total ${st.gpuMs.toFixed(2)} (sum ${sum.toFixed(2)}) | ` + Object.entries(p).map(([k, v]) => `${k} ${v.toFixed(2)}`).join('  '));
}
await browser.close();
