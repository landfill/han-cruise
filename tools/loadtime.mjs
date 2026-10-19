// Cold-load timing in a fresh browser profile (no HTTP cache): time to the enabled start button,
// plus the app's own breakdown (data download+inflate, geometry, GPU pipelines).
import { chromium } from 'playwright-core';
const url = process.argv[2] || 'http://127.0.0.1:4173/';
const runs = [];
for (let i = 0; i < 3; i++) {
  const b = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
  const p = await b.newPage();
  const t0 = Date.now();
  await p.goto(url);
  await p.waitForFunction(() => !document.getElementById('hud').hidden, null, { timeout: 120000 });
  const t = Date.now() - t0;
  const lt = await p.evaluate(() => window.__loadTimes);
  runs.push({ toReady: t, ...lt });
  console.log(`run ${i + 1}: ready ${t} ms  (data ${lt.data.toFixed(0)}, geometry ${lt.geometry.toFixed(0)}, gpu ${lt.gpu.toFixed(0)} ms)`);
  await b.close();
}
