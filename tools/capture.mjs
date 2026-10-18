// Renders the fixed verification scenes in a real Chrome (WebGPU) and saves PNGs + stats.
// Usage: node tools/capture.mjs [--scenes a,b] [--out captures/run] [--w 1920 --h 1080]
//        [--frames 180] [--url http://127.0.0.1:5173/] [--headed]
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { SCENES } from '../src/sim/scenes.js';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const flag = (k) => args.includes('--' + k);
const W = Number(opt('w', 1920)), H = Number(opt('h', 1080));
const OUT = path.resolve(opt('out', 'captures/latest'));
const URL0 = opt('url', 'http://127.0.0.1:5173/');
const FRAMES = Number(opt('frames', 150));
const list = (opt('scenes', '') || Object.keys(SCENES).join(',')).split(',').filter(Boolean);
const extra = opt('params', '');
fs.mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: !flag('headed'), args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist'] });
const results = {};
for (const name of list) {
  const page = await browser.newPage({ viewport: { width: Math.min(W, 1920), height: Math.min(H, 1080) }, deviceScaleFactor: 1 });
  const logs = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  const url = `${URL0}?scene=${name}&w=${W}&h=${H}&ui=0&mute${extra ? '&' + extra : ''}`;
  const t0 = Date.now();
  await page.goto(url);
  try {
    await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
  } catch (e) {
    console.log(name, 'did not become ready'); console.log(logs.join('\n'));
    await page.close();
    continue;
  }
  const tReady = Date.now() - t0;
  await page.waitForFunction((n) => window.__app.frames() > n, FRAMES, { timeout: 120000 });
  const b64 = await page.evaluate(() => window.__app.capture());
  fs.writeFileSync(path.join(OUT, name + '.png'), Buffer.from(b64, 'base64'));
  const stats = await page.evaluate(() => window.__app.stats());
  results[name] = { readyMs: tReady, gpuMs: stats.gpuMs, W: stats.W, H: stats.H, exposure: stats.exposure, errors: stats.errors, load: stats.load, boat: stats.boat };
  const bad = logs.filter((l) => /error|warn|fail/i.test(l));
  console.log(`${name}: ready ${tReady} ms, GPU ${stats.gpuMs.toFixed(2)} ms, ${stats.W}x${stats.H}, exposure ${stats.exposure[0].toFixed(3)}${bad.length ? '\n  ' + bad.slice(0, 12).join('\n  ') : ''}`);
  await page.close();
}
fs.writeFileSync(path.join(OUT, 'stats.json'), JSON.stringify(results, null, 2));
await browser.close();
