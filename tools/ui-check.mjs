// Exercises the real UI flow and the error states in Chrome and saves screenshots:
//   1. loading -> ready start screen        2. start voyage, HUD controls via keyboard and clicks
//   3. scene data fails to load -> error + retry recovers   4. no WebGPU -> unsupported notice
//   5. phone-sized viewport with touch
import fs from 'node:fs';
import { launch } from './browser.mjs';

const OUT = 'captures/ui';
fs.mkdirSync(OUT, { recursive: true });
const URL0 = process.argv[2] || 'http://127.0.0.1:5173/';
const browser = await launch({ headless: true, args: ['--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'] });
const results = [];
const check = (name, cond, detail = '') => { results.push({ name, ok: !!cond, detail }); console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`); };

// 1 + 2: normal flow
{
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(e.message));
  const t0 = Date.now();
  await page.goto(URL0);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/1_loading.png` });
  check('start screen shows a fixed title', (await page.textContent('h1')) === '한강 유람선');
  await page.waitForFunction(() => !document.getElementById('hud').hidden, null, { timeout: 120000 });
  check('voyage starts by itself after loading', true, `${Date.now() - t0} ms`);
  check('starts at golden hour', await page.getAttribute('[data-tod="golden"]', 'aria-pressed') === 'true');
  check('sound is on by default', await page.getAttribute('#soundBtn', 'aria-pressed') === 'true');
  await page.waitForTimeout(900);
  check('start screen fades away', await page.evaluate(() => document.getElementById('start').hidden));
  const blocked = await page.evaluate(() => window.__app.audio().state !== 'running');
  if (blocked) {
    check('muted-autoplay hint shown', await page.isVisible('#soundHint'));
    await page.screenshot({ path: `${OUT}/2_sound_hint.png` });
    await page.mouse.click(800, 450);
    await page.waitForTimeout(400);
    check('first click turns the sound on', await page.evaluate(() => window.__app.audio().state === 'running') && !(await page.isVisible('#soundHint')));
  }
  await page.waitForTimeout(2500);
  const s0 = await page.evaluate(() => window.__app.stats().boat.s);
  await page.waitForTimeout(2000);
  const s1 = await page.evaluate(() => window.__app.stats().boat.s);
  check('boat moves after start', s1 > s0, `${s0.toFixed(1)} -> ${s1.toFixed(1)} m`);
  await page.screenshot({ path: `${OUT}/3_voyage.png` });
  await page.keyboard.press('Space');
  await page.waitForTimeout(300);
  check('Space pauses (button label)', (await page.textContent('#playBtn .lbl')) === '재개');
  const p0 = await page.evaluate(() => window.__app.stats().boat.s);
  await page.waitForTimeout(3000);
  const p1 = await page.evaluate(() => window.__app.stats().boat.s);
  check('boat slows and stops while paused', p1 - p0 < 6, `moved ${(p1 - p0).toFixed(2)} m in 3 s`);
  await page.click('#playBtn');
  check('click resumes', (await page.textContent('#playBtn .lbl')) === '일시정지');
  await page.keyboard.press('Digit4');
  await page.waitForTimeout(3500);
  check('key 4 selects the stern view', (await page.getAttribute('[data-view="stern"]', 'aria-pressed')) === 'true');
  await page.screenshot({ path: `${OUT}/4_stern_view.png` });
  await page.click('[data-tod="night"]');
  await page.waitForTimeout(2500);
  check('time-of-day button switches to night', (await page.getAttribute('[data-tod="night"]', 'aria-pressed')) === 'true');
  await page.screenshot({ path: `${OUT}/5_night.png` });
  const audioBefore = await page.evaluate(() => document.getElementById('soundBtn').textContent);
  await page.keyboard.press('KeyM');
  await page.waitForTimeout(200);
  const audioAfter = await page.evaluate(() => document.getElementById('soundBtn').textContent);
  check('M toggles sound', audioBefore !== audioAfter, `${audioBefore} -> ${audioAfter}`);
  // drag to look around
  const box = await page.locator('#c').boundingBox();
  await page.mouse.move(box.x + 800, box.y + 450);
  await page.mouse.down();
  await page.mouse.move(box.x + 500, box.y + 400, { steps: 8 });
  await page.mouse.up();
  check('pointer drag changes the view (no error)', errors.length === 0, errors.slice(0, 3).join(' | '));
  await page.click('#moreBtn');
  await page.screenshot({ path: `${OUT}/6_settings.png` });
  // focus order reaches the toolbar
  await page.keyboard.press('Tab');
  const focused = await page.evaluate(() => document.activeElement && document.activeElement.id);
  check('keyboard focus reaches controls', !!focused, `focused #${focused}`);
  check('no console errors during the flow', errors.length === 0, errors.slice(0, 3).join(' | '));
  await page.close();
}

// 3: load failure and retry
{
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  let block = true;
  await page.route('**/data/land_hgt.bin.gz', (r) => (block ? r.fulfill({ status: 503, body: 'unavailable' }) : r.continue()));
  await page.goto(URL0);
  await page.waitForSelector('#loadError:not([hidden])', { timeout: 60000 });
  check('load failure shows an error with retry', await page.isVisible('#retryBtn'), await page.textContent('#loadErrorText'));
  await page.screenshot({ path: `${OUT}/7_load_error.png` });
  block = false;
  await page.click('#retryBtn');
  await page.waitForFunction(() => !document.getElementById('hud').hidden, null, { timeout: 120000 });
  check('retry recovers after the failure', true);
  await page.close();
}

// 4: WebGPU unavailable
{
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  await ctx.addInitScript(() => { Object.defineProperty(navigator, 'gpu', { get: () => undefined }); try { delete Navigator.prototype.gpu; } catch (e) {} });
  const page = await ctx.newPage();
  await page.goto(URL0);
  await page.waitForSelector('#unsupported:not([hidden])', { timeout: 20000 });
  check('no WebGPU -> unsupported notice', true, await page.textContent('#unsReason'));
  await page.screenshot({ path: `${OUT}/8_unsupported.png` });
  await ctx.close();
}

// 5: phone viewport with touch
{
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true });
  const page = await ctx.newPage();
  await page.goto(URL0);
  await page.waitForFunction(() => !document.getElementById('hud').hidden, null, { timeout: 120000 });
  await page.waitForTimeout(3000);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  check('phone: no horizontal page overflow', !overflow);
  await page.screenshot({ path: `${OUT}/10_phone_voyage.png` });
  await ctx.close();
}

await browser.close();
fs.writeFileSync(`${OUT}/results.json`, JSON.stringify(results, null, 2));
const fails = results.filter((r) => !r.ok).length;
console.log(`${results.length - fails}/${results.length} checks passed`);
process.exit(fails ? 1 : 0);
