// Deterministic video capture: steps the app frame by frame (fixed dt) and pipes JPEG frames to
// Playwright's bundled ffmpeg (VP9 WebM). Also logs the GPU time per frame and HUD state.
// Usage: node tools/record.mjs --out captures/videos/voyage.webm [--tod golden] [--s0 0] [--s1 4225]
//        [--speed 6] [--fps 30] [--w 1280 --h 720] [--view bow] [--yaw 0 --pitch -3] [--seconds N]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const out = path.resolve(opt('out', 'captures/videos/voyage.webm'));
const tod = opt('tod', 'golden'), s0 = Number(opt('s0', 0)), s1 = Number(opt('s1', 1e9));
const speed = Number(opt('speed', 1)), fps = Number(opt('fps', 30));
const W = Number(opt('w', 1280)), H = Number(opt('h', 720));
const view = opt('view', 'bow'), seconds = Number(opt('seconds', 0));
const yaw = opt('yaw', null), pitch = opt('pitch', null);
const stillsEvery = Number(opt('stills', 0));
const stillDir = out.replace(/\.webm$/, '_stills');
if (stillsEvery) fs.mkdirSync(stillDir, { recursive: true });
fs.mkdirSync(path.dirname(out), { recursive: true });

// Playwright's ffmpeg build only has the file protocol: frames are appended to an MJPEG stream file
// and encoded once recording ends.
const ff = path.join(os.homedir(), 'Library/Caches/ms-playwright/ffmpeg-1013/ffmpeg-mac');
const mjpeg = out.replace(/\.webm$/, '.mjpeg');
const fd = fs.openSync(mjpeg, 'w');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const logs = [];
page.on('console', (m) => { if (/error|warn/i.test(m.type())) logs.push(m.text()); });
const BASE = opt('url', 'http://127.0.0.1:5173/');
await page.goto(`${BASE}?manual&autostart&mute&tod=${tod}&s=${s0}&w=${W}&h=${H}`);
await page.waitForFunction(() => window.__app && window.__app.ready(), null, { timeout: 120000 });
await page.evaluate(([v, y, p, sp]) => { window.__app.setView(v); if (y !== null) window.__app.look(Number(y), Number(p)); window.__app.setSpeed(sp); }, [view, yaw, pitch, speed]);
const dt = 1 / fps;
// warm up (exposure, shadows) without recording
for (let i = 0; i < 30; i++) await page.evaluate((d) => window.__app.step(d), dt);
const total = seconds ? Math.round(seconds * fps) : Infinity;
const t0 = Date.now();
const gpu = [];
const timeline = [];
let n = 0;
while (n < total) {
  const b64 = await page.evaluate((d) => window.__app.stepJpeg(d, 0.88), dt);
  const buf = Buffer.from(b64, 'base64');
  fs.writeSync(fd, buf);
  // keep a still every `--stills` seconds for review (contact sheet)
  if (stillsEvery && n % Math.round(stillsEvery * fps) === 0) fs.writeFileSync(path.join(stillDir, `f${String(n).padStart(5, '0')}.jpg`), buf);
  n++;
  if (n % fps === 0) {
    const st = await page.evaluate(() => ({ ...window.__app.stats(), hud: window.__app.hud() }));
    gpu.push(st.gpuMs);
    timeline.push({ t: +(n / fps).toFixed(1), s: +st.boat.s.toFixed(1), v: +st.boat.v.toFixed(2), gpuMs: +st.gpuMs.toFixed(2), section: st.hud.section, caption: st.hud.caption });
    if (n % (fps * 10) === 0) process.stdout.write(`  ${n / fps}s  s=${st.boat.s.toFixed(0)}  ${st.hud.section}\n`);
    if (st.boat.s >= s1 || (st.boat.v < 0.01 && st.boat.s > 100 && !seconds && n > fps * 20)) break;
  }
}
fs.closeSync(fd);
await browser.close();
const code = await new Promise((r) => spawn(ff, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(fps), '-i', mjpeg,
  '-c:v', 'libvpx-vp9', '-b:v', W >= 1920 ? '4M' : '2M', '-deadline', 'good', '-cpu-used', '4', '-row-mt', '1', out], { stdio: 'inherit' }).on('close', r));
if (code) { console.error('ffmpeg failed', code); process.exit(1); }
fs.unlinkSync(mjpeg);
const meta = { out, frames: n, seconds: n / fps, wallSeconds: (Date.now() - t0) / 1000, tod, speed, W, H, gpuMs: { mean: gpu.reduce((a, b) => a + b, 0) / Math.max(1, gpu.length), max: Math.max(...gpu) }, consoleErrors: logs, timeline };
fs.writeFileSync(out.replace(/\.webm$/, '.json'), JSON.stringify(meta, null, 2));
console.log(`wrote ${out}: ${n} frames, ${(n / fps).toFixed(1)} s video, wall ${meta.wallSeconds.toFixed(0)} s, GPU mean ${meta.gpuMs.mean.toFixed(2)} ms, errors ${logs.length}`);
