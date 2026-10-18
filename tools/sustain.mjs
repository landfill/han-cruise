// Real-use sustained test: 1920x1080 window, default "auto" quality (dynamic resolution), vsync on.
// Logs fps, share of frames over 17.5 ms / 25 ms, render resolution and GPU time every 15 s.
import fs from 'node:fs';
import { chromium } from 'playwright-core';
const URL0 = process.argv[2] || 'http://127.0.0.1:4173/';
const seconds = Number(process.argv[3] || 180);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(`${URL0}?autostart&mute&tod=golden`);
await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
const log = [];
for (let t = 0; t < seconds; t += 15) {
  const r = await page.evaluate(async () => {
    const d = []; let last = performance.now(); const t0 = last;
    await new Promise((res) => { const f = (now) => { d.push(now - last); last = now; if (now - t0 < 15000) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
    const st = window.__app.stats();
    return { fps: d.length / 15, over17: d.filter((x) => x > 17.5).length / d.length, over25: d.filter((x) => x > 25).length, res: `${st.W}x${st.H}`, gpu: st.gpuMs, s: st.boat.s };
  });
  log.push({ t, ...r });
  console.log(`t=${String(t).padStart(3)}s  ${r.fps.toFixed(1)} fps  >17.5ms ${(r.over17 * 100).toFixed(1)}%  >25ms ${r.over25}  render ${r.res}  GPU ${r.gpu.toFixed(1)} ms  s=${r.s.toFixed(0)}`);
}
fs.writeFileSync('captures/sustain.json', JSON.stringify(log, null, 2));
await browser.close();
