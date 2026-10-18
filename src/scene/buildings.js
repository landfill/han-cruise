// Building geometry from OSM footprints (see tools/build-scene.mjs for the packed format).
// Runs in a worker: extruded walls with facade materials, triangulated roofs, landmark
// treatments (63 Building's gold glass and tapering ends, Parc.1's red frames, IFC's chiselled
// crowns, rooftop aviation lights on tall towers). Output: one mesh per 500 m tile for culling.
//
// Wall types go in the vertex colour alpha (read by facade() in src/wgsl/mesh.js):
// 255 windows (balcony front on apartments), 170 apartment back (corridor side),
// 85 gable band carrying the painted block number, 0 blank.
import earcut from 'earcut';
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

const TILE = 1000;
const MATS = ['office', 'residential', 'house', 'glassBlue', 'glassDark', 'gold', 'parc1'];

// Palettes sampled by eye from photographs of Yongsan / Mapo / Yeouido facades (approximate albedo)
const OFFICE_COLS = [[0.52, 0.52, 0.5], [0.6, 0.58, 0.54], [0.42, 0.44, 0.47], [0.47, 0.41, 0.35], [0.64, 0.63, 0.6], [0.34, 0.36, 0.38], [0.55, 0.5, 0.44], [0.3, 0.31, 0.33]];
const RES_COLS = [[0.74, 0.72, 0.68], [0.78, 0.77, 0.74], [0.66, 0.64, 0.6], [0.72, 0.68, 0.6], [0.6, 0.61, 0.62], [0.75, 0.7, 0.62], [0.68, 0.62, 0.55], [0.56, 0.54, 0.52]];
const HOUSE_COLS = [[0.48, 0.34, 0.27], [0.58, 0.5, 0.42], [0.66, 0.64, 0.58], [0.42, 0.37, 0.33], [0.55, 0.42, 0.33]];
// Seoul's flat roofs are often coated with green waterproofing paint
const ROOF_COLS = [[0.4, 0.4, 0.39], [0.36, 0.37, 0.37], [0.14, 0.4, 0.31], [0.18, 0.42, 0.33], [0.46, 0.45, 0.42]];
const GLASS_BLUE = [[0.32, 0.46, 0.55], [0.28, 0.42, 0.45], [0.4, 0.5, 0.55]];
const GLASS_DARK = [[0.22, 0.25, 0.28], [0.26, 0.27, 0.27], [0.2, 0.24, 0.26]];

export function parseBuildings(buffer) {
  const dv = new DataView(buffer);
  let o = 0;
  const n = dv.getInt32(o, true); o += 4;
  const out = [];
  for (let k = 0; k < n; k++) {
    const no = dv.getFloat32(o, true), h = dv.getFloat32(o + 8, true), base = dv.getFloat32(o + 12, true), minH = dv.getFloat32(o + 16, true), floorH = dv.getFloat32(o + 20, true);
    o += 24;
    const seed = dv.getUint32(o, true); o += 4;
    const mat = MATS[dv.getUint8(o)], nr = dv.getUint8(o + 1), flags = dv.getUint16(o + 2, true); o += 4;
    const rings = [];
    for (let r = 0; r < nr; r++) {
      const np = dv.getUint16(o, true), inner = dv.getUint16(o + 2, true) === 1; o += 4;
      const pts = new Float64Array(np * 2);
      for (let i = 0; i < np; i++) { pts[i * 2] = dv.getFloat32(o, true); pts[i * 2 + 1] = dv.getFloat32(o + 4, true); o += 8; }
      rings.push({ inner, pts });
    }
    out.push({ no, h, base, minH, floorH, seed, mat, rings, estimated: !!(flags & 1), landmark: !!(flags & 2) });
  }
  return out;
}

function ringArea(p) {
  let s = 0;
  const n = p.length / 2;
  for (let i = 0; i < n; i++) { const j = (i + 1) % n; s += p[i * 2] * p[j * 2 + 1] - p[j * 2] * p[i * 2 + 1]; }
  return s / 2;
}
const pick = (arr, seed) => arr[seed % arr.length];
const WALL = { WINDOWS: 255, BACK: 170, GABLE_NO: 85, BLANK: 0 };
// gable block number ("101동"): glyphs 1 unit wide with 0.45 gaps, 1.6 units tall. The number
// comes from OSM where tagged; otherwise the shader makes up a three-digit one.
const gableTextW = (no) => { const g = (no ? String(no).length : 3) + 1; return g + (g - 1) * 0.45; };

function principal(p) {
  const n = p.length / 2;
  let mx = 0, mz = 0;
  for (let i = 0; i < n; i++) { mx += p[i * 2]; mz += p[i * 2 + 1]; }
  mx /= n; mz /= n;
  let cxx = 0, czz = 0, cxz = 0;
  for (let i = 0; i < n; i++) { const dx = p[i * 2] - mx, dz = p[i * 2 + 1] - mz; cxx += dx * dx; czz += dz * dz; cxz += dx * dz; }
  const a = 0.5 * Math.atan2(2 * cxz, cxx - czz);
  return { mx, mz, ux: Math.cos(a), uz: Math.sin(a) };
}

export function buildBuildingMeshes(list) {
  const tiles = new Map();
  const lights = []; // aviation obstruction lights: x, y, z
  const stats = { buildings: 0, tris: 0, estimated: 0 };
  for (const b of list) {
    const outer = b.rings.filter((r) => !r.inner);
    if (!outer.length) continue;
    const p0 = outer[0].pts;
    let cx = 0, cz = 0;
    for (let i = 0; i < p0.length; i += 2) { cx += p0[i]; cz += p0[i + 1]; }
    cx /= p0.length / 2; cz /= p0.length / 2;
    const key = Math.floor(cx / TILE) + ',' + Math.floor(cz / TILE);
    if (!tiles.has(key)) tiles.set(key, new MeshBuilder(8192));
    const mb = tiles.get(key);
    stats.buildings++;
    if (b.estimated) stats.estimated++;

    let id = MAT.OFFICE, col = pick(OFFICE_COLS, b.seed);
    if (b.mat === 'residential') { id = MAT.RESIDENTIAL; col = pick(RES_COLS, b.seed); }
    else if (b.mat === 'house') { id = MAT.HOUSE; col = pick(HOUSE_COLS, b.seed); }
    else if (b.mat === 'glassBlue') { id = MAT.CURTAIN; col = pick(GLASS_BLUE, b.seed); }
    else if (b.mat === 'glassDark') { id = MAT.CURTAIN; col = pick(GLASS_DARK, b.seed); }
    else if (b.mat === 'gold') { id = MAT.GOLD; col = [0.8, 0.6, 0.3]; }
    else if (b.mat === 'parc1') { id = MAT.CURTAIN; col = [0.5, 0.56, 0.6]; }
    const y0 = b.base + b.minH, y1 = b.base + b.h;
    // slab-shaped apartment blocks (long and thin): identify their short end walls
    const pr0 = principal(p0);
    let sA = Infinity, sB = -Infinity, wA = Infinity, wB = -Infinity;
    for (let i = 0; i < p0.length; i += 2) {
      const dx = p0[i] - pr0.mx, dz = p0[i + 1] - pr0.mz;
      const sv = dx * pr0.ux + dz * pr0.uz, wv = -dx * pr0.uz + dz * pr0.ux;
      sA = Math.min(sA, sv); sB = Math.max(sB, sv); wA = Math.min(wA, wv); wB = Math.max(wB, wv);
    }
    const lenL = sB - sA, lenW = wB - wA;
    const slab = b.mat === 'residential' && b.h > 18 && lenL / Math.max(lenW, 1) > 2.2;
    const slabEnd = lenW * 1.3 + 1;
    mb.setAux(b.base, b.floorH);
    mb.material(id, col, 0.8, 0, b.seed);

    // 63 Building: its narrow ends lean inwards, the top is ~60 % of the base length
    let taper = null;
    if (b.mat === 'gold') { const pr = principal(p0); taper = { ...pr, top: 0.62 }; }
    // IFC towers: the tops are chiselled off by a sloping glass facet ("crystalline outcrops").
    // The slope direction and angle are approximations from photographs, not drawings.
    let crown = null;
    if (b.landmark && b.mat === 'glassBlue') {
      const dir = (b.seed & 1) ? 1 : -1;
      crown = { u: pr0.ux * dir, w: pr0.uz * dir, s0: dir > 0 ? sA : -sB, len: lenL, drop: Math.min(lenL * 0.55, (y1 - y0) * 0.12) };
    }
    // roof height at a footprint point (y1 except under a crown)
    const topY = (x, z) => {
      if (!crown) return y1;
      const sv = (x - pr0.mx) * crown.u + (z - pr0.mz) * crown.w;
      return y1 - crown.drop * Math.max(0, Math.min(1, (sv - crown.s0) / crown.len));
    };
    const at = (x, z, y) => {
      if (!taper) return [x, z];
      const t = Math.max(0, Math.min(1, (y - y0) / (y1 - y0)));
      const dx = x - taper.mx, dz = z - taper.mz;
      const s = dx * taper.ux + dz * taper.uz, w = -dx * taper.uz + dz * taper.ux;
      const k = 1 + (taper.top - 1) * t;
      return [taper.mx + taper.ux * s * k - taper.uz * w, taper.mz + taper.uz * s * k + taper.ux * w];
    };

    for (const r of b.rings) {
      const p = r.pts, n = p.length / 2;
      const ccw = ringArea(p) > 0;
      const outward = r.inner ? !ccw : ccw;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        const ax = p[i * 2], az = p[i * 2 + 1], bx = p[j * 2], bz = p[j * 2 + 1];
        let nx = bz - az, nz = -(bx - ax);
        if (!outward) { nx = -nx; nz = -nz; }
        if (taper) {
          // split tapered walls into a few bands so the slope reads correctly
          const bands = 6;
          for (let k = 0; k < bands; k++) {
            const ya = y0 + (y1 - y0) * k / bands, yb = y0 + (y1 - y0) * (k + 1) / bands;
            const A0 = at(ax, az, ya), B0 = at(bx, bz, ya), A1 = at(ax, az, yb), B1 = at(bx, bz, yb);
            const ex = B0[0] - A0[0], ez = B0[1] - A0[1];
            const ux = A1[0] - A0[0], uy = yb - ya, uz = A1[1] - A0[1];
            // normal = edge x up, oriented outward
            let fx = -(uy * ez), fy = ez * ux - ex * uz, fz = uy * ex;
            if (fx * nx + fz * nz < 0) { fx = -fx; fy = -fy; fz = -fz; }
            const v0 = mb.vertex(A0[0], ya, A0[1], fx, fy, fz), v1 = mb.vertex(B0[0], ya, B0[1], fx, fy, fz);
            const v2 = mb.vertex(B1[0], yb, B1[1], fx, fy, fz), v3 = mb.vertex(A1[0], yb, A1[1], fx, fy, fz);
            mb.tri(v0, v1, v2); mb.tri(v0, v2, v3);
          }
          continue;
        }
        // apartment slabs: the short end walls are blank and carry the painted block number;
        // of the long walls the south-facing one has the balconies (the usual orientation)
        const edgeLen = Math.hypot(bx - ax, bz - az);
        const blank = slab && edgeLen < slabEnd;
        let wt = WALL.WINDOWS;
        if (blank) wt = WALL.BLANK;
        else if (slab && !r.inner && nz < 0) wt = WALL.BACK;
        mb.col[3] = wt;
        const ta = topY(ax, az), tb = topY(bx, bz);
        let yMid = Math.max(ta, tb);
        let unit = 0;
        if (blank && !r.inner && edgeLen > 8 && y1 - y0 > 25) {
          unit = Math.min(2.4, edgeLen * 0.78 / gableTextW(b.no));
          yMid = y1 - 1.6 * unit - 3.0;
        }
        const v0 = mb.vertex(ax, y0, az, nx, 0, nz), v1 = mb.vertex(bx, y0, bz, nx, 0, nz);
        const v2 = mb.vertex(bx, unit ? yMid : tb, bz, nx, 0, nz), v3 = mb.vertex(ax, unit ? yMid : ta, az, nx, 0, nz);
        mb.tri(v0, v1, v2); mb.tri(v0, v2, v3);
        if (unit) {
          // number band: aux = (band base, glyph unit + 10 * block number), normal.w = horizontal text coordinate
          // (glyph units from the wall centre, along the viewer's right, /8 to fit snorm8)
          const nl = Math.hypot(nx, nz), rx = nz / nl, rz = -nx / nl;
          const mx = (ax + bx) / 2, mz = (az + bz) / 2;
          const tc = (x, z) => Math.round(((x - mx) * rx + (z - mz) * rz) / unit / 8 * 127);
          mb.col[3] = WALL.GABLE_NO;
          mb.setAux(yMid, unit + 10 * b.no);
          const w0 = mb.vertex(ax, yMid, az, nx, 0, nz), w1 = mb.vertex(bx, yMid, bz, nx, 0, nz);
          const w2 = mb.vertex(bx, y1, bz, nx, 0, nz), w3 = mb.vertex(ax, y1, az, nx, 0, nz);
          for (const [vi, x, z] of [[w0, ax, az], [w1, bx, bz], [w2, bx, bz], [w3, ax, az]]) mb.i8[vi * 32 + 15] = tc(x, z);
          mb.tri(w0, w1, w2); mb.tri(w0, w2, w3);
          mb.setAux(b.base, b.floorH);
        }
      }
    }
    mb.col[3] = 255;
    // roof
    const flat = [], holes = [];
    for (const r of [outer[0], ...b.rings.filter((q) => q.inner)]) {
      if (r !== outer[0]) holes.push(flat.length / 2);
      for (let i = 0; i < r.pts.length; i += 2) {
        const q = at(r.pts[i], r.pts[i + 1], y1);
        flat.push(q[0], q[1]);
      }
    }
    const tri = earcut(flat, holes.length ? holes : null, 2);
    if (crown) {
      // the sloping facet is glazed like the walls
      mb.material(id, col, 0.1, 0, b.seed);
    } else mb.material(MAT.ROOF, b.mat === 'house' ? [0.42, 0.33, 0.3] : pick(ROOF_COLS, b.seed >>> 5), 0.9, 0, b.seed);
    const rb = mb.vcount;
    // roof plane normal (tilted under a crown)
    const tn = crown ? [crown.u * crown.drop / crown.len, 1, crown.w * crown.drop / crown.len] : [0, 1, 0];
    for (let i = 0; i < flat.length; i += 2) mb.vertex(flat[i], topY(flat[i], flat[i + 1]), flat[i + 1], tn[0], tn[1], tn[2]);
    for (let i = 0; i < tri.length; i += 3) mb.tri(rb + tri[i], rb + tri[i + 2], rb + tri[i + 1]);
    // Parc.1: red steel frames up the corners and the middle of the long sides
    if (b.mat === 'parc1') {
      mb.material(MAT.STEEL, [0.62, 0.07, 0.05], 0.45, 0.2, b.seed);
      const n = p0.length / 2;
      const pr = principal(p0);
      // bounding rectangle in the principal frame
      let s0 = Infinity, s1 = -Infinity, w0 = Infinity, w1 = -Infinity;
      for (let i = 0; i < n; i++) {
        const dx = p0[i * 2] - pr.mx, dz = p0[i * 2 + 1] - pr.mz;
        const s = dx * pr.ux + dz * pr.uz, w = -dx * pr.uz + dz * pr.ux;
        s0 = Math.min(s0, s); s1 = Math.max(s1, s); w0 = Math.min(w0, w); w1 = Math.max(w1, w);
      }
      const P = (s, w) => [pr.mx + pr.ux * s - pr.uz * w, pr.mz + pr.uz * s + pr.ux * w];
      const posts = [[s0, w0], [s1, w0], [s1, w1], [s0, w1], [(s0 + s1) / 2, w0 - 0.6], [(s0 + s1) / 2, w1 + 0.6]];
      for (const [s, w] of posts) {
        const [x, z] = P(s, w);
        mb.box([x, (y0 + y1 + 8) / 2, z], [pr.ux, 0, pr.uz], [0, 1, 0], [-pr.uz, 0, pr.ux], 1.6, (y1 + 8 - y0) / 2, 1.6);
      }
      // the frame's crowning cross beams
      const [ax, az] = P(s0, w0), [bx, bz] = P(s1, w0), [cx2, cz2] = P(s1, w1), [dx, dz] = P(s0, w1);
      for (const [a, c] of [[[ax, az], [bx, bz]], [[dx, dz], [cx2, cz2]]]) mb.beam([a[0], y1 + 7, a[1]], [c[0], y1 + 7, c[1]], 2.2, 2.4);
    }
    // rooftop structures: stair/elevator cores on apartment blocks, plant rooms on offices
    if (b.h > 20 && b.mat !== 'gold' && b.mat !== 'parc1' && !crown && lenL > 12) {
      const roofCol = b.mat === 'residential' ? pick(RES_COLS, b.seed >>> 3) : [0.45, 0.45, 0.44];
      mb.material(b.mat === 'residential' ? MAT.RESIDENTIAL : MAT.CONCRETE, roofCol, 0.85, 0, b.seed);
      mb.col[3] = 0;
      const P = (sv, wv) => [pr0.mx + pr0.ux * sv - pr0.uz * wv, pr0.mz + pr0.uz * sv + pr0.ux * wv];
      const U3 = [pr0.ux, 0, pr0.uz], W3 = [-pr0.uz, 0, pr0.ux], Y3 = [0, 1, 0];
      if (slab) {
        const n = lenL > 70 ? 3 : 2;
        for (let k = 0; k < n; k++) {
          const sv = sA + lenL * (k + 0.5) / n;
          const [cx2, cz2] = P(sv, (wA + wB) / 2);
          const hh = 3.2 + ((b.seed >>> (k + 2)) & 1) * 1.6;
          mb.box([cx2, y1 + hh / 2, cz2], U3, Y3, W3, 3.5, hh / 2, Math.min(lenW * 0.35, 4));
        }
        // decorative crown frame on the roof edge (common on 1990s-2010s apartment blocks)
        if ((b.seed & 3) === 0) {
          for (const sv of [sA + 1, sB - 1]) {
            const [cx2, cz2] = P(sv, (wA + wB) / 2);
            mb.box([cx2, y1 + 2.5, cz2], U3, Y3, W3, 0.5, 2.5, lenW / 2);
          }
        }
      } else {
        const [cx2, cz2] = P((sA + sB) / 2, (wA + wB) / 2);
        const hh = 3.5 + (b.seed % 3);
        mb.box([cx2, y1 + hh / 2, cz2], U3, Y3, W3, lenL * 0.28, hh / 2, lenW * 0.28);
      }
      mb.col[3] = 255;
    }
    // aviation obstruction lights on tall buildings (roof corners)
    if (b.h > 90) {
      const n = p0.length / 2;
      const step = Math.max(1, Math.floor(n / 4));
      for (let i = 0; i < n; i += step) {
        const q = at(p0[i * 2], p0[i * 2 + 1], y1);
        lights.push(q[0], topY(q[0], q[1]) + 1.2, q[1]);
      }
    }
  }
  const meshes = [];
  for (const [key, mb] of tiles) {
    const r = mb.result();
    stats.tris += r.icount / 3;
    meshes.push({ key, ...r });
  }
  return { meshes, lights: new Float32Array(lights), stats };
}
