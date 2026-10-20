// App entry: WebGPU check, loading with progress/retry, the voyage loop and the Korean UI.
import { loadScene } from './scene/loader.js';
import { buildBridges, buildViaducts } from './scene/bridges.js';
import { buildBoat, BOAT } from './scene/boat.js';
import { buildNodeul, editNodeulTrees } from './scene/nodeul.js';
import { buildLandmarks } from './scene/landmarks.js';
import { buildTreeTemplate } from './scene/trees.js';
import { buildPiers } from './scene/piers.js';
import { buildCarTemplate, buildBusTemplate, CAR_COLOURS, BUS_COLOURS } from './scene/vehicles.js';
import { createRenderer, QUALITY } from './gpu/renderer.js';
import { boatMatrix, D2R, clamp } from './gpu/math.js';
import { Path, Boat } from './sim/route.js';
import { ROUTE_POINTS, LANDMARKS } from './sim/routeData.js';
import { makeEnv, PRESETS } from './sim/environment.js';
import { DeckCamera, FreeCamera } from './sim/camera.js';
import { Traffic } from './sim/traffic.js';
import { CruiseAudio } from './audio/audio.js';
import { routeSections } from './sim/sections.js';
import { SCENES } from './sim/scenes.js';
import { createMinimap } from './ui/minimap.js';
import { createDynRes, createRefreshEstimator, paceDue, periodMs, WINDOW } from './sim/dynres.js';

const qs = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const DEV = qs.has('dev');
const logLines = [];
function log(msg) {
  logLines.push(String(msg));
  console.log(msg);
}
window.__log = logLines;

const state = {
  tod: qs.get('tod') || 'golden',
  sound: !qs.has('mute'),
  info: true,
  paused: false,
  speed: 1,
  reduced: matchMedia('(prefers-reduced-motion: reduce)').matches || qs.get('reduced') === '1',
  quality: qs.get('q') || 'auto',
  started: false,
  free: false,
};

// ---------------------------------------------------------------------------------------------
// WebGPU availability
// ---------------------------------------------------------------------------------------------
async function initGPU() {
  if (!navigator.gpu) return { error: '이 브라우저는 WebGPU를 지원하지 않습니다.' };
  let adapter;
  try { adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); } catch (e) { adapter = null; }
  if (!adapter) return { error: 'WebGPU 어댑터를 찾지 못했습니다. 하드웨어 가속이 꺼져 있거나 GPU가 지원 목록에 없을 수 있습니다.' };
  const want = ['timestamp-query', 'float32-filterable'].filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 << 20) },
  });
  return { adapter, device };
}

function showUnsupported(reason) {
  $('start').hidden = true;
  $('unsupported').hidden = false;
  $('unsReason').textContent = reason;
}

// ---------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------
let gpu, renderer, data, geom, path, boat, deckCam, freeCam, traffic, audio, sections, minimap;
const canvas = $('c');
let context, format;

function setProgress(f, text) {
  $('loadBar').style.width = (f * 100).toFixed(0) + '%';
  document.querySelector('.progress').setAttribute('aria-valuenow', String(Math.round(f * 100)));
  if (text) $('loadText').textContent = text;
}

function buildingsInWorker(buffer) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./scene/worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => { w.terminate(); e.data.ok ? resolve(e.data) : reject(new Error(e.data.error)); };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || '건물 생성 실패')); };
    w.postMessage({ buffer }, [buffer]);
  });
}

async function load() {
  $('loadError').hidden = true;
  try {
    setProgress(0, '그래픽 장치 확인 중');
    if (!gpu) {
      gpu = await initGPU();
      if (gpu.error) { showUnsupported(gpu.error); return; }
      gpu.device.lost.then((info) => {
        log('device lost: ' + info.message);
        if (info.reason !== 'destroyed') fatal('그래픽 장치 연결이 끊어졌습니다. 페이지를 새로 고쳐 주세요. (' + info.message + ')');
      });
      context = canvas.getContext('webgpu');
      format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device: gpu.device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }
    const t0 = performance.now();
    if (!data) data = await loadScene('/data/', (f, t) => setProgress(f * 0.6, t));
    const tData = performance.now();
    setProgress(0.62, '건물 3만여 동 생성 중');
    const b = data.buildings.buffer.slice(data.buildings.byteOffset, data.buildings.byteOffset + data.buildings.byteLength);
    const [buildings] = await Promise.all([buildingsInWorker(b)]);
    setProgress(0.75, '교량·선박 생성 중');
    const bridges = buildBridges(data.meta.bridges);
    const viaducts = buildViaducts(data.meta.viaducts || []);
    const boatGeo = buildBoat({ lite: MOBILE });
    const nodeul = buildNodeul({ meta: data.meta, sdf: data.sdf, complexList: buildings.nodeul, lite: MOBILE });
    if (!data.treesEdited) { data.trees = editNodeulTrees(data.trees, nodeul.treeRow); data.treesEdited = true; }
    boatGeo.dims = { length: BOAT.length, beam: BOAT.beam, draft: BOAT.draft };
    // shadow and reflection passes draw the light build (phones use it everywhere)
    boatGeo.lite = MOBILE ? null : buildBoat({ lite: true }).mesh;
    path = new Path(ROUTE_POINTS, 2);
    geom = {
      path, buildings, bridges, viaducts, boat: boatGeo, nodeul,
      landmarks: buildLandmarks(data.meta.landmarks),
      misc: buildPiers(data.meta.piers),
      treeTemplate: buildTreeTemplate(),
      car: buildCarTemplate(), bus: buildBusTemplate(),
    };
    const tGeom = performance.now();
    setProgress(0.8, '셰이더 준비 중');
    const q = state.quality === 'auto' ? 'high' : state.quality;
    renderer = await createRenderer({ device: gpu.device, context, format, canvas, data, geom, quality: q, reflEvery: Number(qs.get('refl')) || (MOBILE ? 2 : 1), msaa: Number(qs.get('msaa')) || 4, lite: qs.has('lite') ? Number(qs.get('lite')) : (MOBILE ? LITE_MOBILE : 0), log });
    if (MOBILE) {
      dynCtl = createDynRes({ hasTimestamps: renderer.hasTimestamps && !NO_TS });
      // without GPU timestamps, the submit -> work-done time stands in for the GPU time
      renderer.collectDoneTimes(!(renderer.hasTimestamps && !NO_TS));
    }
    const tGpu = performance.now();
    log(`load: data ${(tData - t0).toFixed(0)} ms, geometry ${(tGeom - tData).toFixed(0)} ms, gpu init ${(tGpu - tGeom).toFixed(0)} ms, buildings ${buildings.stats.buildings} (${buildings.stats.estimated} est.), ${(buildings.stats.tris / 1e6).toFixed(2)} M tris`);
    window.__loadTimes = { data: tData - t0, geometry: tGeom - tData, gpu: tGpu - tGeom, total: tGpu - t0 };
    setupSim();
    // warm up: render a few frames behind the start screen so the first visible frame is complete
    setProgress(0.92, '첫 장면 그리는 중');
    for (let i = 0; i < 4; i++) { frame(performance.now(), true); await new Promise((r) => requestAnimationFrame(r)); }
    await gpu.device.queue.onSubmittedWorkDone();
    setProgress(1, '준비 완료');
    start();
  } catch (e) {
    log('load failed: ' + (e && e.stack || e));
    $('loadErrorText').textContent = '불러오기에 실패했습니다: ' + (e && e.message || e);
    $('loadError').hidden = false;
    setProgress(0, '실패');
  }
}

function fatal(msg) {
  $('hud').hidden = true;
  $('start').hidden = false;
  $('loadErrorText').textContent = msg;
  $('loadError').hidden = false;
}

// ---------------------------------------------------------------------------------------------
// simulation setup
// ---------------------------------------------------------------------------------------------
let env;
let simTime = 0;
let shotMode = null;

let passTimes = null;
let passIdx = 0;
let wakeU = 0;

// time at which the boat's centre passed each route sample (drives the age of the wake)
function resetPassTimes(sNow, vEst) {
  passTimes.fill(-1e6);
  passIdx = Math.floor(sNow / path.step);
  for (let i = 0; i <= passIdx && i < passTimes.length; i++) passTimes[i] = simTime - (sNow - i * path.step) / Math.max(vEst, 0.5);
  wakeU = vEst;
}
function advancePassTimes() {
  const k = Math.min(passTimes.length - 1, Math.floor(boat.s / path.step));
  while (passIdx < k) { passIdx++; passTimes[passIdx] = simTime; }
}

function setupSim() {
  passTimes = new Float32Array(path.x.length);
  boat = new Boat(path, { cruise: 4.6, turnSpeed: 2.2, accel: 0.18 });
  sections = routeSections(path, data.meta.bridges);
  try { minimap = createMinimap($('minimap'), { meta: data.meta, sdf: data.sdf, path }); } catch (e) { log('minimap: ' + e.message); }
  deckCam = new DeckCamera(geom.boat.viewpoints, geom.boat.walk);
  freeCam = new FreeCamera();
  const lanes = [...geom.bridges.lanes.map((l) => ({ pts: l.pts, speed: 15 })), ...geom.viaducts.lanes];
  for (const r of data.meta.roads) {
    if (r.bridge) continue;
    const pts = r.pts.map(([x, y, z]) => [x, y, z]);
    const v = r.hw === 'motorway' || r.hw === 'trunk' ? 19 : 13;
    lanes.push({ pts, speed: v });
    if (!r.oneway) {
      // opposite direction, offset to the other side of the road
      const back = [];
      for (let i = pts.length - 1; i >= 0; i--) back.push(pts[i]);
      lanes.push({ pts: back, speed: v });
    }
  }
  traffic = new Traffic(lanes);
  env = makeEnv(state.tod);
  const sc = qs.get('scene');
  if (sc && SCENES[sc]) applyScene(SCENES[sc]);
  else boat.place(Number(qs.get('s')) || 0);
  resetPassTimes(boat.s, boat.v);
}

function applyScene(sc) {
  shotMode = sc;
  state.tod = sc.tod;
  env = makeEnv(sc.tod, sc.env || {});
  simTime = sc.time || 30;
  boat.place(sc.s);
  if (sc.view) deckCam.setView(sc.view, true);
  if (sc.yaw !== undefined) deckCam.yaw = sc.yaw;
  if (sc.lookAt) {
    // aim at a world point from the viewpoint (approximate: from the boat centre)
    const az = Math.atan2(sc.lookAt[0] - boat.x, -(sc.lookAt[1] - boat.z)) / D2R;
    deckCam.yaw = ((az - boat.heading / D2R) % 360 + 540) % 360 - 180;
  }
  if (sc.yawToAz !== undefined) deckCam.yaw = ((sc.yawToAz - boat.heading / D2R) % 360 + 540) % 360 - 180;
  if (sc.pitch !== undefined) deckCam.pitch = sc.pitch;
  if (sc.fov) deckCam.fovY = sc.fov * D2R;
  if (sc.freeRel) {
    // free camera placed relative to the boat: [right, up, back] metres, yaw relative to the bow
    const [r, u, b] = sc.freeRel.pos, h = boat.heading;
    sc.free = { pos: [boat.x + Math.cos(h) * r - Math.sin(h) * b, u, boat.z + Math.sin(h) * r + Math.cos(h) * b], yaw: h / D2R + sc.freeRel.yaw, pitch: sc.freeRel.pitch };
  }
  if (sc.free) { state.free = true; freeCam.pos = sc.free.pos.slice(); freeCam.yaw = sc.free.yaw; freeCam.pitch = sc.free.pitch; if (sc.fov) freeCam.fovY = sc.fov * D2R; }
  state.paused = !!sc.paused;
}

// ---------------------------------------------------------------------------------------------
// voyage
// ---------------------------------------------------------------------------------------------
function start() {
  if (state.started) return;
  state.started = true;
  if (shotMode) state.tod = shotMode.tod;
  state.sound = state.sound && !shotMode;
  env = makeEnv(state.tod, shotMode?.env || {});
  if (qs.get('env')) for (const kv of qs.get('env').split(',')) { const [k, v] = kv.split(':'); env[k] = Number(v); }
  if (qs.get('skip')) env.skip = Object.fromEntries(qs.get('skip').split(',').map((k) => [k, true]));
  leaveStart();
  $('hud').hidden = !!(shotMode && qs.get('ui') !== '1');
  if (qs.get('ui') === '0') $('hud').hidden = true;
  syncUI();
  audio = new CruiseAudio();
  if (state.sound) startAudio();
  canvas.focus();
  resetExposure = true;
}

function leaveStart() {
  const el = $('start');
  if (state.reduced || shotMode) { el.hidden = true; return; }
  el.classList.add('leaving');
  setTimeout(() => { el.hidden = true; el.classList.remove('leaving'); }, 750);
}

// Browsers keep audio muted until the first click, tap or key press on the page: when that is
// the case, show a hint and start the sound on the first interaction.
let unlockArmed = false;
function startAudio() {
  audio.start().catch((e) => log('audio: ' + e));
  if (audio.ctx && audio.ctx.state !== 'running' && !unlockArmed) {
    unlockArmed = true;
    $('soundHint').hidden = false;
    const unlock = () => {
      removeEventListener('pointerdown', unlock, true);
      removeEventListener('keydown', unlock, true);
      unlockArmed = false;
      $('soundHint').hidden = true;
      if (state.sound) audio.start().catch((e) => log('audio: ' + e));
    };
    addEventListener('pointerdown', unlock, true);
    addEventListener('keydown', unlock, true);
  }
}

function setTod(t) {
  if (t === state.tod) return;
  state.tod = t;
  fadeCut(() => { env = makeEnv(t); renderer.resetEnv(); resetExposure = true; });
  syncUI();
}
function setView(k) {
  if (state.free) return;
  if (state.reduced) fadeCut(() => deckCam.setView(k, true));
  else deckCam.setView(k, false);
  syncUI();
}
function fadeCut(fn) {
  const f = $('fade');
  if (state.reduced || shotMode) { fn(); return; }
  f.style.opacity = '1';
  setTimeout(() => { fn(); f.style.opacity = '0'; }, 180);
}
function togglePause() { state.paused = !state.paused; syncUI(); }
function toggleSound() {
  state.sound = !state.sound;
  if (state.sound) startAudio(); else { audio.stop(); $('soundHint').hidden = true; }
  syncUI();
}
function syncUI() {
  $('playBtn').querySelector('.ico').innerHTML = state.paused
    ? '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1l9 5-9 5z" fill="currentColor"/></svg>'
    : '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1" width="3" height="10" fill="currentColor"/><rect x="7" y="1" width="3" height="10" fill="currentColor"/></svg>';
  $('playBtn').querySelector('.lbl').textContent = state.paused ? '재개' : '일시정지';
  $('playBtn').setAttribute('aria-label', state.paused ? '재개' : '일시정지');
  for (const b of document.querySelectorAll('[data-view]')) b.setAttribute('aria-pressed', String(!state.free && b.dataset.view === deckCam?.key));
  for (const b of document.querySelectorAll('[data-tod]')) b.setAttribute('aria-pressed', String(b.dataset.tod === state.tod));
  $('soundBtn').setAttribute('aria-pressed', String(state.sound));
  $('soundBtn').textContent = state.sound ? '소리 켬' : '소리 끔';
  $('infoBtn').setAttribute('aria-pressed', String(state.info));
  $('speedBtn').textContent = state.speed + '×';
  $('reducedChk').checked = state.reduced;
  $('qualitySel').value = state.quality;
}

// ---------------------------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------------------------
const keys = Object.create(null);
let drag = null;
canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, id: e.pointerId }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => {
  if (!drag || drag.id !== e.pointerId) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  const k = (e.pointerType === 'touch' ? 0.22 : 0.15) * (deckCam ? deckCam.fovY / (55 * D2R) : 1);
  if (state.free) freeCam.look(dx * k, -dy * k); else deckCam?.look(dx * k, -dy * k);
});
const endDrag = () => { drag = null; };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('wheel', (e) => { e.preventDefault(); const f = e.deltaY > 0 ? 1.06 : 1 / 1.06; if (state.free) freeCam.fovY = clamp(freeCam.fovY * f, 0.3, 1.5); else deckCam?.zoom(f); }, { passive: false });
addEventListener('keydown', (e) => {
  if (!state.started) return;
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  keys[e.code] = true;
  const views = ['bow', 'port', 'starboard', 'stern', 'foredeck'];
  if (e.code === 'Space' && !state.free) { e.preventDefault(); togglePause(); }
  else if (/^Digit[1-5]$/.test(e.code)) setView(views[Number(e.code.slice(5)) - 1]);
  else if (e.code === 'KeyM') toggleSound();
  else if (e.code === 'KeyI') { state.info = !state.info; syncUI(); }
  else if (e.code === 'KeyT') setTod(['day', 'golden', 'night'][(['day', 'golden', 'night'].indexOf(state.tod) + 1) % 3]);
  else if (e.code === 'KeyF' && DEV) { state.free = !state.free; if (state.free) freeCam.from(lastCam); syncUI(); }
  else if (e.code === 'Backquote') $('devStats').hidden = !$('devStats').hidden;
});
addEventListener('keyup', (e) => { keys[e.code] = false; });
addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

// the caption and settings panel sit above the toolbar, whose height changes when it wraps on phones
new ResizeObserver(([e]) => document.documentElement.style.setProperty('--tb-h', e.target.offsetHeight + 'px')).observe($('toolbar'));
$('retryBtn').addEventListener('click', () => { if (state.started) location.reload(); else load(); });
$('playBtn').addEventListener('click', togglePause);
$('soundBtn').addEventListener('click', toggleSound);
$('infoBtn').addEventListener('click', () => { state.info = !state.info; syncUI(); });
$('speedBtn').addEventListener('click', () => { state.speed = state.speed === 1 ? 2 : state.speed === 2 ? 4 : 1; syncUI(); });
$('moreBtn').addEventListener('click', () => { const s = $('settings'); s.hidden = !s.hidden; $('moreBtn').setAttribute('aria-expanded', String(!s.hidden)); });
$('reducedChk').addEventListener('change', (e) => { state.reduced = e.target.checked; });
$('qualitySel').addEventListener('change', (e) => {
  state.quality = e.target.value; dyn.scale = 1; dyn.ceil = 1;
  if (dynCtl) dynCtl = createDynRes({ hasTimestamps: renderer.hasTimestamps && !NO_TS });
});
// hide the dialog only once the boat is back at the start, or the HUD update re-opens it mid-fade
$('againBtn').addEventListener('click', () => { fadeCut(() => { boat.place(0); boat.finished = false; resetPassTimes(0, 0); $('ended').hidden = true; canvas.focus(); }); state.paused = false; syncUI(); });
for (const b of document.querySelectorAll('[data-view]')) b.addEventListener('click', () => setView(b.dataset.view));
for (const b of document.querySelectorAll('[data-tod]')) b.addEventListener('click', () => setTod(b.dataset.tod));
if (DEV) $('devStats').hidden = false;

// ---------------------------------------------------------------------------------------------
// frame loop
// ---------------------------------------------------------------------------------------------
const boatM = new Float64Array(16);
let last = performance.now();
let lastCam = null;
let resetExposure = true;
let uiTimer = 0;
const fps = { acc: 0, n: 0, t: 0, value: 0, frameMs: [] };
const dyn = { scale: 1, timer: 0, intervals: [], calm: 0, ceil: 1, ceilAge: 0 };
const DYNQ = qs.has('dynq');
const NO_TS = qs.has('nots');   // test the frame-time fallback on a browser that has GPU timestamps
// Phones: resolution levels and frame pacing (src/sim/dynres.js). The screen refresh interval is
// estimated from the rAF callbacks (they come every refresh, rendered or not), from the start of
// loading, while the GPU is still idle; a longer interval later is only believed when the GPU
// cannot be the reason the browser skips callbacks (createRefreshEstimator).
let dynCtl = null;
const refresh = createRefreshEstimator();
const pacing = { get vsync() { return refresh.ms; }, lastRaf: 0, hitches: 0 };
function trackRefresh(now) {
  refresh.add(now - pacing.lastRaf, now / 1000, dynCtl ? dynCtl.state.lastGpu : 0);
  pacing.lastRaf = now;
}
const pace = () => (dynCtl && state.quality === 'auto' ? dynCtl.pace : 1);
const probe = new Float32Array(8);
let boatLightsW = new Float32Array(0);
let frames = 0;

// Phones and tablets (touch-only): a 3x panel at 2x is still ~300 ppi, which 4x MSAA makes
// indistinguishable at arm's length, and it cuts the pixel count by more than half.
// ?mobile=1 / ?mobile=0 forces the choice for testing.
const MOBILE = qs.has('mobile') ? qs.get('mobile') === '1' : matchMedia('(hover: none) and (pointer: coarse)').matches;
const MAX_DPR = MOBILE ? 2 : Infinity;
const PIXEL_BUDGET = Number(qs.get('budget')) || (MOBILE ? 1.4e6 : 1920 * 1080);
// phones compile the reduced shading paths (all LITE flags in src/wgsl/lighting.js); ?lite=N overrides
const LITE_MOBILE = 63;

function renderSize() {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  const fw = Number(qs.get('w')) || 0, fh = Number(qs.get('h')) || 0;
  if (fw && fh) return [fw, fh];
  const cw = canvas.clientWidth * dpr, ch = canvas.clientHeight * dpr;
  const q = QUALITY[state.quality === 'auto' ? 'high' : state.quality];
  // cap the pixel count (~1080p on desktop, ~1.4 MP on touch devices), then apply the quality scale and the dynamic scale
  const cap = Math.min(1, Math.sqrt(PIXEL_BUDGET / (cw * ch)));
  const s = cap * q.scale * (state.quality === 'auto' ? dyn.scale : 1);
  return [Math.max(320, Math.round(cw * s)), Math.max(180, Math.round(ch * s))];
}

// rolling CPU time per frame and frame intervals, shown in the settings panel (readable on a phone)
const diag = { cpu: [], iv: [] };
const roll = (a, x) => { a.push(x); if (a.length > 120) a.shift(); };

function frame(now, warmup = false, dtOverride = 0) {
  const tCpu = performance.now();
  const rawMs = now - last;
  let dt = Math.min(0.1, rawMs / 1000);
  last = now;
  if (!warmup && rawMs < 100) dyn.intervals.push(rawMs);
  if (!warmup && rawMs > 50 && rawMs < 1000) pacing.hitches++;
  if (shotMode || qs.has('fixeddt')) dt = 1 / 60;
  if (dtOverride) dt = dtOverride;
  frames++;
  fps.acc += dt; fps.n++; fps.t += dt;
  fps.frameMs.push(dt * 1000); if (fps.frameMs.length > 240) fps.frameMs.shift();
  if (fps.t > 0.5) { fps.value = fps.n / fps.acc; fps.acc = 0; fps.n = 0; fps.t = 0; }
  simTime += dt;

  const running = state.started && !state.paused;
  const sdt = dt * state.speed;
  boat.update(running ? sdt : 0, running);
  boat.motion(simTime, state.reduced);
  // heave rides the mean water level under the hull (FFT waves are a few cm on the river)
  const wh = renderer.stats.waterHeight;
  const meanH = (wh[0] + wh[1] + wh[2] + wh[3]) / 4;
  boatMatrix(boatM, boat.x, boat.heave + meanH * 0.6, boat.z, boat.heading, boat.pitch, boat.roll);
  const fwd = [Math.sin(boat.heading), -Math.cos(boat.heading)], rt = [Math.cos(boat.heading), Math.sin(boat.heading)];
  probe[0] = boat.x + fwd[0] * 20; probe[1] = boat.z + fwd[1] * 20;
  probe[2] = boat.x - fwd[0] * 20; probe[3] = boat.z - fwd[1] * 20;
  probe[4] = boat.x - rt[0] * 5; probe[5] = boat.z - rt[1] * 5;
  probe[6] = boat.x + rt[0] * 5; probe[7] = boat.z + rt[1] * 5;

  let cam;
  if (state.free) cam = freeCam.update(dt, keys, 0.6);
  else {
    // arrow keys look around
    const kx = (keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0), ky = (keys.ArrowUp ? 1 : 0) - (keys.ArrowDown ? 1 : 0);
    if (kx || ky) deckCam.look(kx * 60 * dt, ky * 40 * dt);
    cam = deckCam.update(dt, boatM, boat, state.reduced);
  }
  // portrait screens: keep at least ~60 degrees of horizontal view
  const aspect = canvas.width / canvas.height;
  if (aspect < 1.1) cam.fovY = Math.min(1.75, Math.max(cam.fovY, 2 * Math.atan(Math.tan(31 * D2R) / aspect)));
  lastCam = cam;

  // boat lights to world
  const bl = geom.boat.lights;
  if (boatLightsW.length !== bl.length) boatLightsW = new Float32Array(bl.length);
  const m = boatM;
  for (let i = 0; i < bl.length; i += 8) {
    const x = bl[i], y = bl[i + 1], z = bl[i + 2];
    boatLightsW[i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    boatLightsW[i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    boatLightsW[i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    for (let k = 3; k < 8; k++) boatLightsW[i + k] = bl[i + k];
  }
  const dynCount = traffic.update(dt, boatLightsW, env.lights > 0.001);
  traffic.bodies(cam.pos, 1800, CAR_COLOURS, BUS_COLOURS);

  advancePassTimes();
  wakeU += (boat.v - wakeU) * Math.min(1, dt / 6);   // the wake pattern adapts to speed changes over seconds
  env.boat.x = boat.x; env.boat.z = boat.z; env.boat.heading = boat.heading; env.boat.v = boat.v; env.boat.yawRate = boat.yawRate;
  env.boat.s = boat.s; env.boat.wakeU = wakeU;
  const [w, h] = renderSize();
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  renderer.render({
    cam, env, time: simTime, dt, boatMatrix: boatM, width: w, height: h, probe,
    dynLights: { data: traffic.data, count: dynCount }, resetExposure, passTimes,
    vehicles: { cars: traffic.cars, carCount: traffic.carCount, buses: traffic.buses, busCount: traffic.busCount },
  });
  resetExposure = false;

  if (!warmup) {
    uiTimer += dt;
    if (uiTimer > 0.25) { uiTimer = 0; updateHUD(cam); }
    if (audio && state.sound) audio.update(dt, { speed: boat.v, underBridge: underBridge(cam.pos), paused: state.paused, night: env.darkness, reduced: state.reduced });
    autoQuality(dt);
    if (rawMs < 100) roll(diag.iv, rawMs);
    roll(diag.cpu, performance.now() - tCpu);
  }
}

function underBridge(p) {
  for (const o of geom.bridges.occluders) {
    const [ax, az] = o.a, [bx, bz] = o.b;
    const L = Math.hypot(bx - ax, bz - az), ux = (bx - ax) / L, uz = (bz - az) / L;
    const s = (p[0] - ax) * ux + (p[2] - az) * uz, n = -(p[0] - ax) * uz + (p[2] - az) * ux;
    if (s > 0 && s < L && Math.abs(n) < o.width / 2 + 6) return 1 - Math.abs(n) / (o.width / 2 + 6);
  }
  return 0;
}

function autoQuality(dt) {
  // ?dynq: let the dynamic resolution run in a fixed scene too (measuring what it settles at)
  if (state.quality !== 'auto' || (shotMode && !DYNQ)) { dyn.intervals.length = 0; return; }
  dyn.timer += dt;
  if (dynCtl) {
    if (dyn.timer < WINDOW) return;
    dyn.timer = 0;
    const iv = dyn.intervals;
    dyn.intervals = [];
    dynCtl.update({ gpuMs: renderer.stats.gpuMs, intervals: iv, doneMs: renderer.takeDoneTimes(), vsyncMs: pacing.vsync, now: performance.now() / 1000 });
    dyn.scale = dynCtl.scale;
    return;
  }
  if (dyn.timer < 1.5) return;
  dyn.timer = 0;
  const iv = dyn.intervals;
  dyn.intervals = [];
  if (!renderer.hasTimestamps || NO_TS) { frameTimeQuality(iv); return; }
  const g = renderer.stats.gpuMs;
  if (!g) return;
  // keep GPU time under ~14 ms (60 Hz with headroom): pixel count scales roughly with time
  if (g > 14.5) dyn.scale = Math.max(0.6, dyn.scale * Math.sqrt(13 / g));
  else if (g < 10.5 && dyn.scale < 1) dyn.scale = Math.min(1, dyn.scale * 1.06);
}

// Without GPU timestamps (Safari): count frames that missed a 60 Hz refresh (> 25 ms apart).
// Step down quickly when frames are dropped; step back up slowly, and not straight back to a
// scale that just failed (that ceiling expires after 30 s, e.g. when the scene gets lighter).
function frameTimeQuality(iv) {
  if (iv.length < 20) return;
  let missed = 0;
  for (const x of iv) if (x > 25) missed++;
  const r = missed / iv.length;
  dyn.ceilAge += 1.5;
  if (dyn.ceilAge > 30) dyn.ceil = 1;
  if (r > 0.08) {
    dyn.ceil = dyn.scale; dyn.ceilAge = 0; dyn.calm = 0;
    dyn.scale = Math.max(0.6, dyn.scale * (r > 0.3 ? 0.85 : 0.93));
  } else if (r < 0.01 && ++dyn.calm >= 3) {
    dyn.calm = 0;
    dyn.scale = Math.max(dyn.scale, Math.min(1, dyn.ceil * 0.97, dyn.scale * 1.04));
  }
}

function updateHUD(cam) {
  const s = boat.s;
  const sec = sections.find((x) => s >= x.s0 && s < x.s1) || sections[sections.length - 1];
  $('sectionName').textContent = sec.name;
  const remain = (path.length - s) / 4.3;
  $('eta').textContent = boat.finished ? '' : `· 남은 거리 ${((path.length - s) / 1000).toFixed(1)}km · 약 ${Math.max(1, Math.round(remain / 60))}분`;
  $('routeFill').style.width = (100 * s / path.length).toFixed(1) + '%';
  if (minimap) minimap.update(boat);
  if (boat.finished && $('ended').hidden && state.started && !shotMode) $('ended').hidden = false;
  // landmark caption: the most central visible landmark within range
  let best = null, bestScore = 0;
  if (state.info) {
    for (const lm of LANDMARKS) {
      const dx = lm.pos[0] - cam.pos[0], dz = lm.pos[1] - cam.pos[2];
      const d = Math.hypot(dx, dz);
      if (d > lm.range) continue;
      const dy = lm.height * 0.5 - cam.pos[1];
      const L = Math.hypot(dx, dy, dz);
      const c = (dx * cam.fwd[0] + dy * cam.fwd[1] + dz * cam.fwd[2]) / L;
      // must be inside the horizontal field of view (with a margin)
      const halfX = Math.min(28 * D2R, Math.atan(Math.tan(cam.fovY / 2) * canvas.width / canvas.height) * 0.85);
      if (c < Math.cos(halfX)) continue;
      const score = c * (1.5 - d / lm.range);
      if (score > bestScore) { bestScore = score; best = lm; }
    }
  }
  const cap = $('caption');
  if (best) {
    if ($('capName').textContent !== best.name) { $('capName').textContent = best.name; $('capText').textContent = best.text; }
    cap.hidden = false;
  } else cap.hidden = true;
  const st = renderer.stats;
  const ms = fps.frameMs.slice().sort((a, b) => a - b);
  const p95 = ms[Math.floor(ms.length * 0.95)] || 0;
  const cpu = diag.cpu.slice().sort((a, b) => a - b);
  const cpuMed = cpu[cpu.length >> 1] || 0, cpuMax = cpu[cpu.length - 1] || 0;
  // a frame is late when it comes half a refresh after its paced time (25 ms at 60 fps)
  const late = MOBILE ? periodMs(pace(), pacing.vsync) + pacing.vsync * 0.5 : 25;
  const missed = diag.iv.length ? diag.iv.filter((x) => x > late).length / diag.iv.length : 0;
  $('perfLine').textContent = `${st.W}×${st.H} (${dyn.scale.toFixed(2)}) · ${fps.value.toFixed(0)} fps`
    + (MOBILE ? ` (목표 ${Math.round(1000 / periodMs(pace(), pacing.vsync))}, 화면 ${Math.round(1000 / pacing.vsync)} Hz)` : '')
    + ` · 끊긴 프레임 ${(missed * 100).toFixed(0)}% · 50ms 넘는 멈춤 ${pacing.hitches}회 · 해상도 변경 ${st.resizes}회 · CPU ${cpuMed.toFixed(1)}/${cpuMax.toFixed(0)} ms`
    + (renderer.hasTimestamps ? ` · GPU ${st.gpuMs.toFixed(1)} ms` : '');
  if (!$('devStats').hidden) {
    $('devStats').textContent = `${fps.value.toFixed(0)} fps  p95 ${p95.toFixed(1)} ms\nGPU ${st.gpuMs.toFixed(2)} ms\n${st.W}x${st.H} scale ${dyn.scale.toFixed(2)}\nwater ${st.waterNodes} land ${st.landNodes} meshes ${st.meshesDrawn}\ns ${boat.s.toFixed(0)} m  v ${boat.v.toFixed(2)} m/s\ncam ${cam.pos.map((v) => v.toFixed(1)).join(', ')}\nexp ${st.exposure[0].toFixed(3)} avgL ${st.exposure[1].toExponential(2)}\nsun ${env.sunElev.toFixed(1)}° az ${env.sunAzim.toFixed(0)}°`;
  }
}

const MANUAL = qs.has('manual');   // frames are stepped by tools/record.mjs
function loop(now) {
  requestAnimationFrame(loop);
  if (MOBILE) trackRefresh(now);
  if (!renderer || !state.started || MANUAL) { last = now; return; }
  // phones: render on every n-th screen refresh (fixed scenes render every callback, unless the
  // dynamic resolution runs in them: ?dynq)
  if (MOBILE && (!shotMode || DYNQ) && !paceDue(now, last, pace(), pacing.vsync)) return;
  frame(now);
}

// automation hooks (used by tools/capture.mjs)
window.__app = {
  ready: () => !!renderer && state.started,
  frames: () => frames,
  capture: async () => {
    const img = await renderer.capture();
    const c = new OffscreenCanvas(img.width, img.height);
    c.getContext('2d').putImageData(img, 0, 0);
    const blob = await c.convertToBlob({ type: 'image/png' });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = '';
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(s);
  },
  stats: () => ({ ...renderer.stats, waterHeight: Array.from(renderer.stats.waterHeight), exposure: Array.from(renderer.stats.exposure), fps: fps.value, frameMs: fps.frameMs.slice(), errors: renderer.errors.slice(), dynScale: dyn.scale, pace: pace(), vsync: pacing.vsync, hitches: pacing.hitches, dynEstimate: dynCtl?.state.estimate, boat: { s: boat.s, v: boat.v, x: boat.x, z: boat.z, heading: boat.heading }, load: window.__loadTimes, log: logLines.slice(-30) }),
  // render one frame with a fixed dt and return it as a JPEG (base64)
  stepJpeg: async (dt, quality = 0.9) => {
    const p = renderer.capture();
    frame(performance.now(), false, dt);
    const img = await p;
    const c = new OffscreenCanvas(img.width, img.height);
    c.getContext('2d').putImageData(img, 0, 0);
    const buf = new Uint8Array(await (await c.convertToBlob({ type: 'image/jpeg', quality })).arrayBuffer());
    let s = '';
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(s);
  },
  step: (dt) => frame(performance.now(), false, dt),
  hud: () => ({ section: document.getElementById('sectionName').textContent, caption: document.getElementById('caption').hidden ? '' : document.getElementById('capName').textContent }),
  setBoat: (s) => { boat.place(s); resetPassTimes(boat.s, boat.v); },
  setSpeed: (k) => { state.speed = k; },
  setPaused: (p) => { state.paused = p; },
  setView: (k) => deckCam.setView(k, true),
  look: (yaw, pitch) => { deckCam.yaw = yaw; deckCam.pitch = pitch; },
  setTod: (t) => { state.tod = t; env = makeEnv(t); renderer.resetEnv(); resetExposure = true; },
  envSet: (o) => Object.assign(env, o),
  pathLength: () => path.length,
  sections: () => sections,
  clickStart: () => start(),
  audio: () => ({ state: audio?.ctx?.state || 'none', on: !!audio?.on, level: audio ? audio.level() : 0 }),
};

requestAnimationFrame(loop);
void PRESETS;
load();
