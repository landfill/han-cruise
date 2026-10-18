// Builds the runtime scene data in public/data/ from the raw downloads in data-raw/.
//   river.sdf    signed distance to the Han River waterline (m, + on land), 4 m grid
//   land.hgt     land height above the normal water level (m), 8 m grid, banks shaped by profile
//   far.hgt      background terrain (mountains) 64 m grid
//   landclass    surface class (park, wood, road, path, rail, sand, urban), 4 m grid
//   buildings    footprints, heights, materials (geometry is generated in the browser)
//   scene.json   bridges (axes, spans, deck profile), landmarks, roads, lamps, trees, route
// All coordinates are local metres: x = east, z = south, y = up; y = 0 is the normal
// Han River water level at Yeouido (about EL 2.5 m, held by the Singok weir).
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { PNG } from 'pngjs';

const RAW = path.resolve('data-raw');
const OUT = path.resolve('public/data');
fs.mkdirSync(OUT, { recursive: true });
const WATER_EL = 2.5;

// ---- projection ---------------------------------------------------------------------------
const LAT0 = 37.525, LON0 = 126.945;
const KX = 111320 * Math.cos((LAT0 * Math.PI) / 180), KZ = 110574;
const toXZ = (lat, lon) => [(lon - LON0) * KX, -(lat - LAT0) * KZ];
const toLatLon = (x, z) => [LAT0 - z / KZ, LON0 + x / KX];

const load = (n) => JSON.parse(fs.readFileSync(path.join(RAW, n + '.json'), 'utf8')).elements;
function geoms(e) {
  const out = [];
  if (e.geometry) out.push({ role: 'outer', pts: e.geometry.filter(Boolean).map((p) => toXZ(p.lat, p.lon)) });
  for (const m of e.members || []) {
    if (m.geometry) out.push({ role: m.role || 'outer', pts: m.geometry.filter(Boolean).map((p) => toXZ(p.lat, p.lon)) });
  }
  return out;
}

// ---- grids ----------------------------------------------------------------------------------
class Grid {
  constructor(x0, z0, w, h, cell, Arr = Float32Array) {
    Object.assign(this, { x0, z0, w, h, cell });
    this.a = new Arr(w * h);
  }
  ix(x) { return Math.floor((x - this.x0) / this.cell); }
  iz(z) { return Math.floor((z - this.z0) / this.cell); }
  cx(i) { return this.x0 + (i + 0.5) * this.cell; }
  cz(j) { return this.z0 + (j + 0.5) * this.cell; }
  sample(x, z) { // bilinear on cell centres
    const fx = (x - this.x0) / this.cell - 0.5, fz = (z - this.z0) / this.cell - 0.5;
    const i = Math.max(0, Math.min(this.w - 2, Math.floor(fx))), j = Math.max(0, Math.min(this.h - 2, Math.floor(fz)));
    const tx = Math.max(0, Math.min(1, fx - i)), tz = Math.max(0, Math.min(1, fz - j));
    const a = this.a, w = this.w;
    return (a[j * w + i] * (1 - tx) + a[j * w + i + 1] * tx) * (1 - tz) + (a[(j + 1) * w + i] * (1 - tx) + a[(j + 1) * w + i + 1] * tx) * tz;
  }
}

// Supercover line rasterisation (marks every cell the segment touches)
function rasterLine(g, ax, az, bx, bz, fn) {
  const steps = Math.ceil(Math.hypot(bx - ax, bz - az) / (g.cell * 0.35)) + 1;
  for (let k = 0; k <= steps; k++) {
    const t = k / steps;
    const i = g.ix(ax + (bx - ax) * t), j = g.iz(az + (bz - az) * t);
    if (i >= 0 && j >= 0 && i < g.w && j < g.h) fn(j * g.w + i, i, j);
  }
}
function rasterThick(g, pts, width, fn) {
  const r = width / 2, rc = Math.ceil(r / g.cell) + 1;
  for (let k = 0; k < pts.length - 1; k++) {
    const [ax, az] = pts[k], [bx, bz] = pts[k + 1];
    const minI = g.ix(Math.min(ax, bx) - r), maxI = g.ix(Math.max(ax, bx) + r);
    const minJ = g.iz(Math.min(az, bz) - r), maxJ = g.iz(Math.max(az, bz) + r);
    if (maxI < 0 || maxJ < 0 || minI >= g.w || minJ >= g.h) continue;
    const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz || 1e-9;
    for (let j = Math.max(0, minJ); j <= Math.min(g.h - 1, maxJ); j++) {
      for (let i = Math.max(0, minI); i <= Math.min(g.w - 1, maxI); i++) {
        const px = g.cx(i), pz = g.cz(j);
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / L2));
        if (Math.hypot(px - ax - dx * t, pz - az - dz * t) <= r) fn(j * g.w + i);
      }
    }
    void rc;
  }
}
// Even-odd scanline polygon fill (rings = array of point arrays)
function rasterPoly(g, rings, fn) {
  let minZ = Infinity, maxZ = -Infinity;
  for (const r of rings) for (const p of r) { minZ = Math.min(minZ, p[1]); maxZ = Math.max(maxZ, p[1]); }
  const j0 = Math.max(0, g.iz(minZ)), j1 = Math.min(g.h - 1, g.iz(maxZ));
  const xs = [];
  for (let j = j0; j <= j1; j++) {
    const z = g.cz(j);
    xs.length = 0;
    for (const r of rings) {
      for (let k = 0; k < r.length; k++) {
        const a = r[k], b = r[(k + 1) % r.length];
        if ((a[1] <= z && b[1] > z) || (b[1] <= z && a[1] > z)) xs.push(a[0] + ((z - a[1]) / (b[1] - a[1])) * (b[0] - a[0]));
      }
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const i0 = Math.max(0, Math.ceil((xs[k] - g.x0) / g.cell - 0.5)), i1 = Math.min(g.w - 1, Math.floor((xs[k + 1] - g.x0) / g.cell - 0.5));
      for (let i = i0; i <= i1; i++) fn(j * g.w + i);
    }
  }
}

// 1-D squared Euclidean distance transform (Felzenszwalb & Huttenlocher)
function edt1d(f, n, d, v, z) {
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s;
    while (true) {
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      if (s <= z[k]) { k--; continue; }
      break;
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}
function edt(mask, w, h, target) { // distance (cells) from each cell to nearest cell with mask == target
  const INF = 1e20, out = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] === target ? 0 : INF;
  const n = Math.max(w, h), f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = out[y * w + x];
    edt1d(f, h, d, v, z);
    for (let y = 0; y < h; y++) out[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = out[y * w + x];
    edt1d(f, w, d, v, z);
    for (let x = 0; x < w; x++) out[y * w + x] = Math.sqrt(d[x]);
  }
  return out;
}

// ---- river polylines & water mask -------------------------------------------------------
const riverWays = load('river').map((e) => e.geometry.map((p) => toXZ(p.lat, p.lon)));
// segment spatial hash for exact distances near the bank
const SEG_CELL = 64;
const segHash = new Map();
for (const w of riverWays) {
  for (let k = 0; k < w.length - 1; k++) {
    const [ax, az] = w[k], [bx, bz] = w[k + 1];
    const i0 = Math.floor(Math.min(ax, bx) / SEG_CELL), i1 = Math.floor(Math.max(ax, bx) / SEG_CELL);
    const j0 = Math.floor(Math.min(az, bz) / SEG_CELL), j1 = Math.floor(Math.max(az, bz) / SEG_CELL);
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const key = i + ',' + j;
      if (!segHash.has(key)) segHash.set(key, []);
      segHash.get(key).push([ax, az, bx, bz]);
    }
  }
}
function exactBankDist(x, z, maxR) {
  let best = Infinity;
  const r = Math.ceil(maxR / SEG_CELL);
  const ci = Math.floor(x / SEG_CELL), cj = Math.floor(z / SEG_CELL);
  for (let j = cj - r; j <= cj + r; j++) for (let i = ci - r; i <= ci + r; i++) {
    const segs = segHash.get(i + ',' + j);
    if (!segs) continue;
    for (const [ax, az, bx, bz] of segs) {
      const dx = bx - ax, dz = bz - az, L2 = dx * dx + dz * dz || 1e-9;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / L2));
      best = Math.min(best, Math.hypot(x - ax - dx * t, z - az - dz * t));
    }
  }
  return best;
}

// Seeds known to be open river (checked against the OSM polygon plot)
const WATER_SEEDS = [[0, 0], [-1500, -900], [600, 300], [2500, 1300], [-3000, -1700]];

function waterMask(g) {
  const WALL = 2, WATER = 1;
  const m = new Uint8Array(g.w * g.h);
  for (const w of riverWays) for (let k = 0; k < w.length - 1; k++) rasterLine(g, w[k][0], w[k][1], w[k + 1][0], w[k + 1][1], (idx) => { m[idx] = WALL; });
  const stack = [];
  for (const [x, z] of WATER_SEEDS) {
    const i = g.ix(x), j = g.iz(z);
    if (i >= 0 && j >= 0 && i < g.w && j < g.h && m[j * g.w + i] === 0) stack.push(j * g.w + i);
  }
  while (stack.length) {
    const idx = stack.pop();
    if (m[idx] !== 0) continue;
    m[idx] = WATER;
    const i = idx % g.w, j = (idx - i) / g.w;
    if (i > 0) stack.push(idx - 1);
    if (i < g.w - 1) stack.push(idx + 1);
    if (j > 0) stack.push(idx - g.w);
    if (j < g.h - 1) stack.push(idx + g.w);
  }
  // boundary cells: water if the cell centre is inside by majority of neighbours
  const out = new Uint8Array(g.w * g.h);
  for (let j = 0; j < g.h; j++) for (let i = 0; i < g.w; i++) {
    const idx = j * g.w + i;
    if (m[idx] !== WALL) { out[idx] = m[idx] === WATER ? 1 : 0; continue; }
    let wc = 0, lc = 0;
    for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) {
      const ii = i + a, jj = j + b;
      if (ii < 0 || jj < 0 || ii >= g.w || jj >= g.h) continue;
      const v = m[jj * g.w + ii];
      if (v === WATER) wc++; else if (v === 0) lc++;
    }
    out[idx] = wc > lc ? 1 : 0;
  }
  const frac = out.reduce((s, v) => s + v, 0) / out.length;
  console.log(`water mask ${g.w}x${g.h} @${g.cell}m: ${(frac * 100).toFixed(1)}% water`);
  return out;
}

function signedDistance(g, mask, exactR) {
  const dIn = edt(mask, g.w, g.h, 0);   // for water cells: distance to land
  const dOut = edt(mask, g.w, g.h, 1);  // for land cells: distance to water
  const sdf = new Float32Array(g.w * g.h);
  for (let j = 0; j < g.h; j++) for (let i = 0; i < g.w; i++) {
    const idx = j * g.w + i;
    const water = mask[idx] === 1;
    let d = water ? -(dIn[idx] - 0.5) * g.cell : (dOut[idx] - 0.5) * g.cell;
    if (Math.abs(d) < exactR) {
      const e = exactBankDist(g.cx(i), g.cz(j), exactR + g.cell * 2);
      if (e < Infinity) d = water ? -e : e;
    }
    sdf[idx] = d;
  }
  return sdf;
}

// ---- DEM ----------------------------------------------------------------------------------
function loadTerrarium(z) {
  const dir = path.join(RAW, 'terrarium', String(z));
  const tiles = new Map();
  for (const f of fs.readdirSync(dir)) {
    const [x, y] = f.replace('.png', '').split('_').map(Number);
    tiles.set(x + ',' + y, PNG.sync.read(fs.readFileSync(path.join(dir, f))));
  }
  const n = 2 ** z;
  return (lat, lon) => {
    const fx = ((lon + 180) / 360) * n * 256;
    const r = (lat * Math.PI) / 180;
    const fy = ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n * 256;
    const px = fx - 0.5, py = fy - 0.5;
    const x0 = Math.floor(px), y0 = Math.floor(py), tx = px - x0, ty = py - y0;
    const get = (X, Y) => {
      const t = tiles.get(Math.floor(X / 256) + ',' + Math.floor(Y / 256));
      if (!t) return NaN;
      const i = ((Y & 255) * 256 + (X & 255)) * 4;
      return t.data[i] * 256 + t.data[i + 1] + t.data[i + 2] / 256 - 32768;
    };
    return (get(x0, y0) * (1 - tx) + get(x0 + 1, y0) * tx) * (1 - ty) + (get(x0, y0 + 1) * (1 - tx) + get(x0 + 1, y0 + 1) * tx) * ty;
  };
}
// Separable square min/max filter (for a morphological opening of the DEM)
function morph(a, w, h, r, useMax) {
  const tmp = new Float32Array(a.length);
  const op = useMax ? Math.max : Math.min;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    let v = a[j * w + i];
    for (let k = Math.max(0, i - r); k <= Math.min(w - 1, i + r); k++) v = op(v, a[j * w + k]);
    tmp[j * w + i] = v;
  }
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    let v = tmp[j * w + i];
    for (let k = Math.max(0, j - r); k <= Math.min(h - 1, j + r); k++) v = op(v, tmp[k * w + i]);
    a[j * w + i] = v;
  }
}
function blur(a, w, h, radiusCells) { // separable box blur x3 ~ gaussian
  const tmp = new Float32Array(a.length);
  const pass = (src, dst, horiz) => {
    const r = radiusCells;
    if (horiz) {
      for (let j = 0; j < h; j++) {
        let s = 0, c = 0;
        for (let i = -r; i <= r; i++) { const ii = Math.min(w - 1, Math.max(0, i)); s += src[j * w + ii]; c++; }
        for (let i = 0; i < w; i++) {
          dst[j * w + i] = s / c;
          const add = Math.min(w - 1, i + r + 1), sub = Math.max(0, i - r);
          s += src[j * w + add] - src[j * w + sub];
        }
      }
    } else {
      for (let i = 0; i < w; i++) {
        let s = 0, c = 0;
        for (let j = -r; j <= r; j++) { const jj = Math.min(h - 1, Math.max(0, j)); s += src[jj * w + i]; c++; }
        for (let j = 0; j < h; j++) {
          dst[j * w + i] = s / c;
          const add = Math.min(h - 1, j + r + 1), sub = Math.max(0, j - r);
          s += src[add * w + i] - src[sub * w + i];
        }
      }
    }
  };
  for (let k = 0; k < 3; k++) { pass(a, tmp, true); pass(tmp, a, false); }
}

// Bank profile: height (m above water) as a function of signed distance d from the waterline.
// Riprap revetment, a promenade, the riverside park terrace, then the city behind the levee.
// Mirrors bankProfile() in src/wgsl/terrain.js — keep the two in sync.
function bankProfile(d) {
  if (d < 0) return Math.max(-6, d * 0.45 - 0.25);
  if (d < 5) return -0.25 + (d / 5) * 2.85;           // riprap slope ~1:1.8
  if (d < 14) return 2.6 + (d - 5) * 0.01;            // promenade
  if (d < 40) return 2.7 + ((d - 14) / 26) ** 2 * 1.3; // grass bank up to the park terrace
  return 4.0;
}

// ---- outputs helpers ---------------------------------------------------------------------
function writeBin(name, typed, meta) {
  const buf = Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength);
  const gz = zlib.gzipSync(buf, { level: 9 });
  fs.writeFileSync(path.join(OUT, name + '.bin.gz'), gz);
  console.log(`${name}: ${(buf.length / 1e6).toFixed(1)} MB -> ${(gz.length / 1e6).toFixed(2)} MB gz`);
  return { file: name + '.bin.gz', ...meta };
}

// =============================================================================================
console.time('total');
const scene = { projection: { lat0: LAT0, lon0: LON0, kx: KX, kz: KZ, waterElevation: WATER_EL } };

// Corridor (fine) grid: SDF + land classes
const COR = new Grid(-5000, -4000, 2750, 1875, 4);
const corMask = waterMask(COR);
const corSdf = signedDistance(COR, corMask, 48);

// Height grid (8 m) over the wider area
const HG = new Grid(-7400, -6600, 2300, 1650, 8);
const hMask = waterMask(HG);
const hSdf = signedDistance(HG, hMask, 48);
const dem14 = loadTerrarium(14);
const dem = new Float32Array(HG.w * HG.h);
for (let j = 0; j < HG.h; j++) for (let i = 0; i < HG.w; i++) {
  const [lat, lon] = toLatLon(HG.cx(i), HG.cz(j));
  const v = dem14(lat, lon);
  dem[j * HG.w + i] = Number.isFinite(v) ? v - WATER_EL : 6;
}
// SRTM is a surface model: in Yeouido and Yongsan it includes the tower blocks. A morphological
// opening (erode then dilate, 85 m) removes features narrower than ~170 m while keeping hills such as
// Namsan; the blur then smooths the remaining SRTM noise.
morph(dem, HG.w, HG.h, 10, false);
morph(dem, HG.w, HG.h, 10, true);
blur(dem, HG.w, HG.h, 3);
for (let k = 0; k < HG.a.length; k++) {
  const d = hSdf[k];
  const p = bankProfile(d);
  if (d <= 40) { HG.a[k] = p; continue; }
  // levee/city: blend from the park terrace into the (smoothed) DEM
  const t = Math.min(1, Math.max(0, (d - 90) / 160));
  const ts = t * t * (3 - 2 * t);
  HG.a[k] = p + Math.max(0, dem[k] - p) * ts + (ts > 0 ? 0 : 0);
}
const landH = (x, z) => {
  if (x > COR.x0 && z > COR.z0 && x < COR.x0 + COR.w * COR.cell && z < COR.z0 + COR.h * COR.cell) {
    const d = COR.sample(x, z);
    if (d < 40) return bankProfile(d);
  }
  return HG.sample(x, z);
};
COR.a = corSdf;

// Far terrain (64 m) for the mountains ringing Seoul
const FG = new Grid(-32000, -32000, 1000, 1000, 64);
const dem11 = loadTerrarium(11);
for (let j = 0; j < FG.h; j++) for (let i = 0; i < FG.w; i++) {
  const [lat, lon] = toLatLon(FG.cx(i), FG.cz(j));
  const v = dem11(lat, lon);
  FG.a[j * FG.w + i] = Number.isFinite(v) ? Math.max(v - WATER_EL, -3) : 0;
}

// ---- land classes ---------------------------------------------------------------------------
const LC = { URBAN: 0, GRASS: 1, WOOD: 2, ROAD: 3, SAND: 4, RAIL: 5, PATH: 6, PAVED: 7 };
const cls = new Uint8Array(COR.w * COR.h);
// the Han riverside parks run almost continuously along both banks
for (let k = 0; k < cls.length; k++) { const d = corSdf[k]; if (d > 12 && d < 160) cls[k] = LC.GRASS; else if (d >= 4 && d <= 12) cls[k] = LC.PAVED; }
for (const e of load('landuse')) {
  const t = e.tags || {};
  let c = -1;
  if (t.natural === 'wood' || t.landuse === 'forest') c = LC.WOOD;
  else if (t.natural === 'sand' || t.natural === 'beach') c = LC.SAND;
  else if (t.leisure === 'park' || t.landuse === 'grass' || t.landuse === 'recreation_ground' || t.natural === 'scrub') c = LC.GRASS;
  if (c < 0) continue;
  const rings = geoms(e).filter((g) => g.pts.length > 2).map((g) => g.pts);
  if (!rings.length) continue;
  rasterPoly(COR, rings, (idx) => { if (corSdf[idx] > 3 && (c !== LC.GRASS || cls[idx] !== LC.WOOD)) cls[idx] = c; });
}
const ROAD_W = { motorway: 26, trunk: 22, primary: 18, secondary: 13, motorway_link: 9, trunk_link: 9 };
const roadsOut = [];
for (const e of load('roads')) {
  const t = e.tags || {};
  const pts = geoms(e)[0]?.pts;
  if (!pts || pts.length < 2) continue;
  if (t.railway) {
    if (!t.tunnel) rasterThick(COR, pts, 8, (idx) => { if (corSdf[idx] > 2) cls[idx] = LC.RAIL; });
    continue;
  }
  if (t.highway === 'cycleway' || t.highway === 'footway' || t.highway === 'path') {
    rasterThick(COR, pts, t.highway === 'cycleway' ? 3.2 : 2.4, (idx) => { if (corSdf[idx] > 4) cls[idx] = LC.PATH; });
    continue;
  }
  const w = ROAD_W[t.highway] || 10;
  if (t.tunnel) continue;
  if (!t.bridge) rasterThick(COR, pts, w, (idx) => { if (corSdf[idx] > 3) cls[idx] = LC.ROAD; });
  if (['motorway', 'trunk', 'primary'].includes(t.highway)) {
    roadsOut.push({ hw: t.highway, oneway: t.oneway === 'yes' || t.highway === 'motorway', lanes: Number(t.lanes) || (t.highway === 'motorway' ? 3 : 2), bridge: !!t.bridge, name: t.name || '', pts });
  }
}

// ---- bridges --------------------------------------------------------------------------------
// Structural types and spans: see docs/VERIFICATION.md for sources and which values are estimates.
const BRIDGE_SPECS = {
  875036745: { key: 'wonhyo', name: '원효대교', type: 'boxgirder', deckW: 20, span: 100, deckTop: 15.5, depthPier: 5.5, depthMid: 2.6, pier: 'v' },
  880050989: { key: 'rail_a', name: '한강철교 A선', type: 'truss_warren', deckW: 7, span: 62, deckTop: 11.2, trussH: 8.5 },
  880050990: { key: 'rail_b', name: '한강철교 B선', type: 'truss_warren', deckW: 7, span: 62, deckTop: 11.2, trussH: 8.5 },
  880050987: { key: 'rail_c', name: '한강철교 C선', type: 'truss_cont', deckW: 7.5, span: 62, deckTop: 11.2, trussH: 9.5 },
  880050988: { key: 'rail_d', name: '한강철교 D선', type: 'truss_cont', deckW: 7.5, span: 62, deckTop: 11.2, trussH: 9.5 },
  872068121: { key: 'mapo', name: '마포대교', type: 'boxgirder', deckW: 25, span: 90, deckTop: 14.5, depthPier: 4.5, depthMid: 2.8, pier: 'twin' },
  875036744: { key: 'seogang', name: '서강대교', type: 'boxgirder', deckW: 30, span: 80, deckTop: 15.0, depthPier: 3.5, depthMid: 2.5, pier: 'twin', archSpan: 150 },
  870733632: { key: 'dongjak', name: '동작대교', type: 'boxgirder', deckW: 40, span: 85, deckTop: 14.5, depthPier: 3.2, depthMid: 2.6, pier: 'twin', arches: 3 },
  1025707744: { key: 'dangsan', name: '당산철교', type: 'truss_cont', deckW: 11, span: 60, deckTop: 12.0, trussH: 7 },
  872068120: { key: 'yanghwa', name: '양화대교', type: 'boxgirder', deckW: 28, span: 70, deckTop: 14.0, depthPier: 3.0, depthMid: 2.4, pier: 'twin', arches: 2 },
};
const HANGANG_REL = 11970731;

function pca(pts) {
  const n = pts.length;
  let mx = 0, mz = 0;
  for (const p of pts) { mx += p[0]; mz += p[1]; }
  mx /= n; mz /= n;
  let cxx = 0, czz = 0, cxz = 0;
  for (const p of pts) { const dx = p[0] - mx, dz = p[1] - mz; cxx += dx * dx; czz += dz * dz; cxz += dx * dz; }
  const a = 0.5 * Math.atan2(2 * cxz, cxx - czz);
  const ux = Math.cos(a), uz = Math.sin(a);
  let s0 = Infinity, s1 = -Infinity, w0 = Infinity, w1 = -Infinity;
  for (const p of pts) {
    const s = (p[0] - mx) * ux + (p[1] - mz) * uz, w = -(p[0] - mx) * uz + (p[1] - mz) * ux;
    s0 = Math.min(s0, s); s1 = Math.max(s1, s); w0 = Math.min(w0, w); w1 = Math.max(w1, w);
  }
  const cw = (w0 + w1) / 2;
  const cx = mx - uz * cw, cz = mz + ux * cw;
  return { a: [cx + ux * s0, cz + uz * s0], b: [cx + ux * s1, cz + uz * s1], width: w1 - w0 };
}
const sdfAt = (x, z) => {
  if (x > COR.x0 && z > COR.z0 && x < COR.x0 + COR.w * COR.cell && z < COR.z0 + COR.h * COR.cell) return COR.sample(x, z);
  return HG.sample(x, z) > 0.5 ? 50 : -50;
};
function buildBridge(spec, a, b) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const ux = (b[0] - a[0]) / L, uz = (b[1] - a[1]) / L;
  // find the over-water interval along the axis
  let s0 = null, s1 = null;
  for (let s = 0; s <= L; s += 1) {
    if (sdfAt(a[0] + ux * s, a[1] + uz * s) < 0) { if (s0 === null) s0 = s; s1 = s; }
  }
  if (s0 === null) return null;
  // main piers: equal spans across the water, bank piers set back onto the revetment
  const wL = s1 - s0;
  const n = Math.max(1, Math.round(wL / spec.span));
  const piers = [];
  for (let k = 0; k <= n; k++) piers.push(s0 + (wL * k) / n);
  // land approach piers every ~35 m out to the ends
  for (let s = s0 - 35; s > 25; s -= 35) piers.push(s);
  for (let s = s1 + 35; s < L - 25; s += 35) piers.push(s);
  piers.sort((p, q) => p - q);
  // deck profile: level over the river with a gentle crest, ramps down to the ends
  const prof = [];
  for (let s = 0; s <= L + 0.01; s += 10) {
    const x = a[0] + ux * s, z = a[1] + uz * s;
    const ground = landH(x, z);
    const t = Math.min(1, Math.max(0, (s - s0) / Math.max(wL, 1)));
    const crest = spec.deckTop + 0.9 * Math.sin(Math.PI * t);
    const endDist = Math.min(s, L - s);
    const ramp = Math.min(1, endDist / 140);
    const end = Math.max(ground, 0) + 1.0;
    prof.push(+(end + (crest - end) * (ramp * ramp * (3 - 2 * ramp))).toFixed(2));
  }
  return { ...spec, a: a.map((v) => +v.toFixed(2)), b: b.map((v) => +v.toFixed(2)), length: +L.toFixed(1), water: [+s0.toFixed(1), +s1.toFixed(1)], piers: piers.map((p) => +p.toFixed(1)), profStep: 10, profile: prof };
}
const bridgesOut = [];
const bridgeEls = load('bridges');
// The four Hangang Railway Bridge lines stand side by side: their piers line up across the river.
// Pier positions are generated for line A and projected onto the other three axes.
function alignPiers(ref, br) {
  const [ax, az] = ref.a, [bx, bz] = ref.b;
  const L = Math.hypot(bx - ax, bz - az), ux = (bx - ax) / L, uz = (bz - az) / L;
  const [cx, cz] = br.a, [dx, dz] = br.b;
  const L2 = Math.hypot(dx - cx, dz - cz), vx = (dx - cx) / L2, vz = (dz - cz) / L2;
  br.piers = ref.piers.map((s) => {
    const x = ax + ux * s, z = az + uz * s;
    return +((x - cx) * vx + (z - cz) * vz).toFixed(1);
  }).filter((s) => s > 5 && s < L2 - 5);
}
for (const e of bridgeEls) {
  const spec = BRIDGE_SPECS[e.id];
  if (!spec) continue;
  const pts = geoms(e).flatMap((g) => g.pts);
  const ax = pca(pts);
  const br = buildBridge(spec, ax.a, ax.b);
  if (br) { br.osmId = e.id; bridgesOut.push(br); console.log(`bridge ${spec.name}: L=${br.length} water ${br.water} spans=${br.piers.length}`); }
}
{
  const ref = bridgesOut.find((b) => b.key === 'rail_a');
  for (const b of bridgesOut) if (b.key.startsWith('rail_') && b !== ref) alignPiers(ref, b);
}
// Hangang Bridge: two parallel decks (relation members); arches on the Nodeul-Noryangjin side
{
  const rel = bridgeEls.find((e) => e.id === HANGANG_REL);
  const parts = geoms(rel).filter((g) => g.pts.length > 3);
  parts.forEach((g, k) => {
    const ax = pca(g.pts);
    const br = buildBridge({ key: 'hangang' + k, name: '한강대교', type: 'hangang', deckW: Math.min(ax.width, 22), span: 60, deckTop: 14.0, depthPier: 2.6, depthMid: 2.2, pier: 'twin' }, ax.a, ax.b);
    if (br) { br.osmId = rel.id; bridgesOut.push(br); console.log(`bridge 한강대교 part ${k}: L=${br.length} water ${br.water}`); }
  });
}

// ---- buildings ----------------------------------------------------------------------------
const LANDMARKS = {
  64989671: { key: '63', name: '63빌딩', mat: 'gold' },
  874639189: { key: 'parc1', name: '파크원 타워', mat: 'parc1' },
  621752526: { key: 'ifc1', name: 'IFC 서울', mat: 'glassBlue' },
  621752527: { key: 'ifc2', name: 'IFC 서울', mat: 'glassBlue' },
  621752528: { key: 'ifc3', name: 'IFC 서울', mat: 'glassBlue' },
  681805210: { key: 'fki', name: 'FKI 타워', mat: 'glassDark' },
};
const DUPLICATE_OF = { 463384209: 64989671 }; // "63스퀘어" outline duplicates the 63 Building way
function num(v) { const n = parseFloat(String(v || '').replace(',', '.')); return Number.isFinite(n) ? n : null; }
function ringArea(r) { let s = 0; for (let k = 0; k < r.length; k++) { const a = r[k], b = r[(k + 1) % r.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; }
function hash32(n) { n = (n ^ 61) ^ (n >>> 16); n = Math.imul(n, 9); n ^= n >>> 4; n = Math.imul(n, 0x27d4eb2d); n ^= n >>> 15; return n >>> 0; }

const seen = new Set();
const blds = [];
let estimated = 0;
for (const f of ['buildings_wide', 'buildings_corridor', 'landmarks']) {
  for (const e of load(f)) {
    if (seen.has(e.type + e.id)) continue;
    seen.add(e.type + e.id);
    const t = e.tags || {};
    if (!t.building || DUPLICATE_OF[e.id]) continue;
    if (['roof', 'construction', 'ruins', 'no'].includes(t.building)) continue;
    let rings = geoms(e).filter((g) => g.pts.length > 3);
    if (!rings.length) continue;
    // drop closing duplicate point
    rings = rings.map((g) => {
      const p = g.pts.slice();
      if (p.length > 1 && p[0][0] === p[p.length - 1][0] && p[0][1] === p[p.length - 1][1]) p.pop();
      return { role: g.role, pts: p };
    }).filter((g) => g.pts.length >= 3);
    const outer = rings.filter((g) => g.role !== 'inner');
    if (!outer.length) continue;
    const area = Math.abs(outer.reduce((s, g) => s + ringArea(g.pts), 0));
    if (area < 12) continue;
    let h = num(t.height);
    const lv = num(t['building:levels']);
    const isEst = h === null && lv === null;
    const seed = hash32(e.id);
    const kind = t.building;
    let floorH = ['apartments', 'residential', 'house', 'detached', 'dormitory'].includes(kind) ? 2.9 : 3.8;
    if (h === null && lv !== null) h = lv * floorH + 1.5;
    if (h === null) {
      estimated++;
      // footprint heuristics for untagged buildings (documented as estimates)
      const [cx, cz] = outer[0].pts.reduce((s, p) => [s[0] + p[0], s[1] + p[1]], [0, 0]).map((v) => v / outer[0].pts.length);
      const ax = pca(outer[0].pts);
      const len = Math.hypot(ax.b[0] - ax.a[0], ax.b[1] - ax.a[1]);
      const slab = len > 40 && ax.width > 9 && ax.width < 20 && len / ax.width > 2.8;
      if (kind === 'apartments' || slab) { h = (12 + (seed % 14)) * 2.9 + 1.5; floorH = 2.9; }
      else if (kind === 'house' || kind === 'detached') h = 7 + (seed % 3) * 2.9;
      else if (area > 1500) h = 18 + (seed % 5) * 3.8;
      else if (area > 400) h = 11 + (seed % 4) * 3.3;
      else h = 6.5 + (seed % 3) * 3.0;
      void cx; void cz;
    }
    // floating terminals / restaurants on the river: the pier pontoons are modelled separately
    {
      const c0 = outer[0].pts.reduce((a, p) => [a[0] + p[0], a[1] + p[1]], [0, 0]).map((v) => v / outer[0].pts.length);
      if (sdfAt(c0[0], c0[1]) < 0) {
        if (t.man_made === 'pier') continue;
        if (isEst) h = 7.5;
      }
    }
    const minH = num(t.min_height) || 0;
    // base: lowest ground under the footprint (buildings never float)
    let base = Infinity;
    for (const p of outer[0].pts) base = Math.min(base, landH(p[0], p[1]));
    if (!Number.isFinite(base)) continue;
    base = Math.max(base, 2.6) - 0.6;
    const lm = LANDMARKS[e.id] || null;
    let mat;
    if (lm) mat = lm.mat;
    else if (['apartments', 'residential', 'dormitory'].includes(kind) || floorH < 3) mat = 'residential';
    else if (h > 70 || (['commercial', 'office'].includes(kind) && h > 40)) mat = (seed & 3) === 0 ? 'glassDark' : (seed & 3) === 1 ? 'glassBlue' : 'office';
    else if (kind === 'house' || kind === 'detached') mat = 'house';
    else mat = 'office';
    blds.push({ id: e.id, rings, h: Math.max(h, 3), minH, base, mat, floorH, seed, est: isEst, lm: lm?.key || null });
  }
}
console.log(`buildings: ${blds.length} (${estimated} with estimated height)`);

// pack: header int32 count, then per building: f32[x0,z0,h,base,minH,floorH], u32 seed, u8 mat, u8 nRings, u16 pad,
// per ring: u16 nPts, u16 inner flag, f32 pts...
const MATS = ['office', 'residential', 'house', 'glassBlue', 'glassDark', 'gold', 'parc1'];
const parts = [];
let bytes = 4;
for (const b of blds) {
  bytes += 6 * 4 + 4 + 4;
  for (const r of b.rings) bytes += 4 + r.pts.length * 8;
}
const bb = Buffer.alloc(bytes);
let o = 0;
bb.writeInt32LE(blds.length, o); o += 4;
for (const b of blds) {
  for (const v of [0, 0, b.h, b.base, b.minH, b.floorH]) { bb.writeFloatLE(v, o); o += 4; }
  bb.writeUInt32LE(b.seed, o); o += 4;
  bb.writeUInt8(MATS.indexOf(b.mat), o); bb.writeUInt8(b.rings.length, o + 1); bb.writeUInt16LE((b.est ? 1 : 0) | (b.lm ? 2 : 0), o + 2); o += 4;
  for (const r of b.rings) {
    bb.writeUInt16LE(r.pts.length, o); bb.writeUInt16LE(r.role === 'inner' ? 1 : 0, o + 2); o += 4;
    for (const p of r.pts) { bb.writeFloatLE(p[0], o); bb.writeFloatLE(p[1], o + 4); o += 8; }
  }
}
void parts;
fs.writeFileSync(path.join(OUT, 'buildings.bin.gz'), zlib.gzipSync(bb, { level: 9 }));
console.log(`buildings.bin: ${(bb.length / 1e6).toFixed(1)} MB`);

// ---- trees ----------------------------------------------------------------------------------
// Riverside park trees (jittered grid in grass/wood, kept off paths, roads and the promenade)
const trees = [];
let rs = 12345;
const rnd = () => { rs = (Math.imul(rs, 1664525) + 1013904223) >>> 0; return rs / 4294967296; };
const TREE_CELL = 11;
for (let z = COR.z0; z < COR.z0 + COR.h * COR.cell; z += TREE_CELL) {
  for (let x = COR.x0; x < COR.x0 + COR.w * COR.cell; x += TREE_CELL) {
    const px = x + rnd() * TREE_CELL, pz = z + rnd() * TREE_CELL;
    const i = COR.ix(px), j = COR.iz(pz);
    const k = j * COR.w + i;
    const c = cls[k], d = corSdf[k];
    if (d < 18 || d > 420) continue;
    let p = 0;
    if (c === LC.WOOD) p = 0.85;
    else if (c === LC.GRASS) p = d < 120 ? 0.32 : 0.22;
    else continue;
    // clumps: low-frequency modulation
    const clump = 0.5 + 0.5 * Math.sin(px * 0.013 + Math.sin(pz * 0.011) * 2.0) * Math.cos(pz * 0.017 - px * 0.004);
    if (rnd() > p * (0.35 + 0.9 * clump)) continue;
    // keep clear of paths/roads in a 6 m radius
    let clear = true;
    for (let b = -1; b <= 1 && clear; b++) for (let a = -1; a <= 1; a++) {
      const cc = cls[(j + b) * COR.w + (i + a)];
      if (cc === LC.PATH || cc === LC.ROAD || cc === LC.RAIL || cc === LC.PAVED) { clear = false; break; }
    }
    if (!clear) continue;
    trees.push(px, pz, landH(px, pz), 0.75 + rnd() * 0.6, rnd());
  }
}
for (const e of load('piers')) {
  if (e.tags?.natural !== 'tree_row') continue;
  const pts = geoms(e)[0]?.pts || [];
  for (let k = 0; k < pts.length - 1; k++) {
    const L = Math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]);
    for (let s = 0; s < L; s += 8) {
      const px = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * s / L, pz = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * s / L;
      if (sdfAt(px, pz) < 6) continue;
      trees.push(px, pz, landH(px, pz), 0.8 + rnd() * 0.4, rnd());
    }
  }
}
console.log(`trees: ${trees.length / 5}`);
scene.trees = writeBin('trees', new Float32Array(trees), { count: trees.length / 5, stride: 5 });

// ---- lamps (street lights) -----------------------------------------------------------------
const lamps = [];
function lampsAlong(pts, spacing, heightFn, side) {
  let acc = 0;
  for (let k = 0; k < pts.length - 1; k++) {
    const [ax, az] = pts[k], [bx, bz] = pts[k + 1];
    const L = Math.hypot(bx - ax, bz - az);
    const nx = -(bz - az) / L, nz = (bx - ax) / L;
    for (let s = (spacing - acc) % spacing; s < L; s += spacing) {
      const x = ax + (bx - ax) * s / L + nx * side, z = az + (bz - az) * s / L + nz * side;
      lamps.push(+x.toFixed(1), +heightFn(x, z).toFixed(2), +z.toFixed(1));
    }
    acc = (acc + L) % spacing;
  }
}
for (const r of roadsOut) {
  if (r.bridge) continue;
  if (r.pts.every(([x, z]) => sdfAt(x, z) > 900)) continue; // only roads near the river are seen as rows of lamps
  const w = ROAD_W[r.hw] / 2 + 0.5;
  lampsAlong(r.pts, 38, (x, z) => landH(x, z) + 10, w);
  if (!r.oneway) lampsAlong(r.pts, 38, (x, z) => landH(x, z) + 10, -w);
}
for (const e of load('roads')) {
  const t = e.tags || {};
  if (t.highway !== 'cycleway' && t.highway !== 'footway') continue;
  const pts = geoms(e)[0]?.pts || [];
  if (!pts.length || pts.some(([x, z]) => sdfAt(x, z) > 140 || sdfAt(x, z) < 3)) continue;
  lampsAlong(pts, 30, (x, z) => landH(x, z) + 4.5, 1.5);
}
console.log(`lamps: ${lamps.length / 3}`);
scene.lamps = writeBin('lamps', new Float32Array(lamps), { count: lamps.length / 3 });

// roads for night traffic: simplified, with heights (bridge decks handled at runtime)
function simplify(pts, tol) {
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
  const st = [[0, pts.length - 1]];
  while (st.length) {
    const [i0, i1] = st.pop();
    let best = -1, bi = -1;
    const [ax, az] = pts[i0], [bx, bz] = pts[i1];
    const L = Math.hypot(bx - ax, bz - az) || 1e-9;
    for (let i = i0 + 1; i < i1; i++) {
      const d = Math.abs((bx - ax) * (az - pts[i][1]) - (ax - pts[i][0]) * (bz - az)) / L;
      if (d > best) { best = d; bi = i; }
    }
    if (best > tol) { keep[bi] = 1; st.push([i0, bi], [bi, i1]); }
  }
  return pts.filter((_, i) => keep[i]);
}
scene.roads = roadsOut
  .filter((r) => r.pts.some(([x, z]) => sdfAt(x, z) < 700))
  .map((r) => ({ hw: r.hw, oneway: r.oneway, lanes: r.lanes, bridge: r.bridge, name: r.name, pts: simplify(r.pts, 1.5).map(([x, z]) => [+x.toFixed(1), +(r.bridge ? -1 : landH(x, z) + 0.15).toFixed(2), +z.toFixed(1)]) }));
console.log(`roads: ${scene.roads.length}`);

// ---- elevated roads (viaducts and ramps along the banks) -----------------------------------
// OSM bridge=yes road ways that are not river crossings (those are modelled as bridges above).
// Deck height: ground + 6 m per layer; free ends ramp down to the ground over ~70 m.
{
  const VIA_HW = { motorway: 24, trunk: 20, primary: 17, secondary: 12, motorway_link: 8, trunk_link: 8, primary_link: 8 };
  const ways = load('roads').filter((e) => {
    const t = e.tags || {};
    return t.bridge && !t.railway && VIA_HW[t.highway] && e.geometry && e.geometry.length >= 2;
  });
  const endCount = new Map();
  for (const e of ways) for (const id of [e.nodes[0], e.nodes[e.nodes.length - 1]]) endCount.set(id, (endCount.get(id) || 0) + 1);
  scene.viaducts = [];
  for (const e of ways) {
    const pts = e.geometry.map((p) => toXZ(p.lat, p.lon));
    const water = pts.filter(([x, z]) => sdfAt(x, z) < 0).length;
    if (water > pts.length * 0.3) continue;              // river crossing: modelled as a bridge
    if (pts.every(([x, z]) => sdfAt(x, z) > 1200)) continue; // not visible from the river
    const layer = Math.max(1, Math.min(3, Number(e.tags.layer) || 1));
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const L = cum[cum.length - 1];
    const freeA = endCount.get(e.nodes[0]) === 1, freeB = endCount.get(e.nodes[e.nodes.length - 1]) === 1;
    const out = pts.map(([x, z], i) => {
      const g = landH(x, z);
      let h = 6.0 * layer;
      const r = 70;
      if (freeA) h *= Math.min(1, cum[i] / r);
      if (freeB) h *= Math.min(1, (L - cum[i]) / r);
      return [+x.toFixed(1), +(g + Math.max(h, 0.2)).toFixed(2), +z.toFixed(1), +g.toFixed(2)];
    });
    scene.viaducts.push({ hw: e.tags.highway, width: VIA_HW[e.tags.highway], oneway: e.tags.oneway === 'yes' || e.tags.highway === 'motorway', pts: out });
  }
  console.log(`viaducts: ${scene.viaducts.length}`);
}

// ---- piers / terminal ---------------------------------------------------------------------
scene.piers = [];
for (const e of load('piers')) {
  const t = e.tags || {};
  if (t.man_made !== 'pier') continue;
  const g = geoms(e)[0];
  if (!g || g.pts.length < 3) continue;
  const [cx, cz] = g.pts.reduce((s, p) => [s[0] + p[0], s[1] + p[1]], [0, 0]).map((v) => v / g.pts.length);
  if (Math.hypot(cx, cz) > 4500) continue;
  scene.piers.push({ name: t.name || '', floating: t.floating === 'yes', building: !!t.building, pts: g.pts.map(([x, z]) => [+x.toFixed(2), +z.toFixed(2)]) });
}

// ---- landmarks (named, for captions and special geometry) -------------------------------
const lmPos = (lat, lon) => { const [x, z] = toXZ(lat, lon); return [+x.toFixed(1), +landH(x, z).toFixed(1), +z.toFixed(1)]; };
scene.landmarks = {
  nSeoulTower: { pos: lmPos(37.551169, 126.988227), height: 236.7 },
  lotteWorldTower: { pos: lmPos(37.512539, 127.102540), height: 554.5 },
};
{
  const [x, , z] = scene.landmarks.nSeoulTower.pos;
  const [lat, lon] = toLatLon(x, z);
  scene.landmarks.nSeoulTower.demEl = +(dem14(lat, lon)).toFixed(1);
  // the tower stands on the summit: use the unsmoothed DEM there
  scene.landmarks.nSeoulTower.pos[1] = +(scene.landmarks.nSeoulTower.demEl - WATER_EL).toFixed(1);
}

// ---- grids out --------------------------------------------------------------------------
const q16 = (arr, scale) => { const o = new Int16Array(arr.length); for (let i = 0; i < arr.length; i++) o[i] = Math.max(-32767, Math.min(32767, Math.round(arr[i] * scale))); return o; };
scene.sdf = writeBin('river_sdf', q16(corSdf, 20), { x0: COR.x0, z0: COR.z0, w: COR.w, h: COR.h, cell: COR.cell, scale: 1 / 20 });
scene.landclass = writeBin('landclass', cls, { x0: COR.x0, z0: COR.z0, w: COR.w, h: COR.h, cell: COR.cell });
scene.height = writeBin('land_hgt', q16(HG.a, 50), { x0: HG.x0, z0: HG.z0, w: HG.w, h: HG.h, cell: HG.cell, scale: 1 / 50 });
scene.far = writeBin('far_hgt', q16(FG.a, 10), { x0: FG.x0, z0: FG.z0, w: FG.w, h: FG.h, cell: FG.cell, scale: 1 / 10 });
scene.buildings = { file: 'buildings.bin.gz', count: blds.length, estimated, mats: MATS };
scene.bridges = bridgesOut;
scene.attribution = 'Map data © OpenStreetMap contributors (ODbL 1.0). Elevation: Mapzen/AWS Terrain Tiles (SRTM, NASA; and others).';
fs.writeFileSync(path.join(OUT, 'scene.json'), JSON.stringify(scene));
console.log('scene.json', (fs.statSync(path.join(OUT, 'scene.json')).size / 1e6).toFixed(2), 'MB');

// debug overview image of the processed grids
{
  const W = COR.w / 2, H = COR.h / 2;
  const png = new PNG({ width: W, height: H });
  const pal = [[120, 115, 108], [96, 140, 70], [50, 95, 50], [60, 60, 64], [200, 185, 140], [110, 90, 80], [170, 160, 150], [150, 145, 140]];
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const k = (j * 2) * COR.w + i * 2;
    const d = corSdf[k];
    let c = d < 0 ? [40, 70 + Math.max(-60, d) * 0.3, 100] : pal[cls[k]];
    const o2 = (j * W + i) * 4;
    png.data[o2] = c[0]; png.data[o2 + 1] = c[1]; png.data[o2 + 2] = c[2]; png.data[o2 + 3] = 255;
  }
  fs.mkdirSync('captures/debug', { recursive: true });
  fs.writeFileSync('captures/debug/landclass.png', PNG.sync.write(png));
}
console.timeEnd('total');
