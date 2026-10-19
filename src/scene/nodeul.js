// Nodeul island (노들섬) close-up detail, built at load from the committed scene data:
//  - the music & culture complex (2019, 9,747 m², three storeys or fewer, on the Yongsan side west
//    of the Hangang Bridge): its OSM footprints are drawn as stepped masses - a glazed ground
//    floor, white upper storeys set back, lawn roofs with railings - instead of plain boxes.
//    Storey heights and set-backs are estimates from photographs (see docs/VERIFICATION.md).
//  - lamp posts along the perimeter promenade (traced on the river distance field),
//  - the bridge abutments where the Hangang Bridge decks land on the island,
//  - trees: an open lawn west of the complex and a row of trees along the promenade.
import earcut from 'earcut';
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';
import { bridgeFrame } from './bridges.js';

// window (scene metres) that holds the complex footprints; buildings in it taller than 5 m are
// taken out of the general building mesh and rebuilt here
export const NODEUL_COMPLEX = { x0: 1080, x1: 1300, z0: 715, z1: 880, minH: 5 };
export const isNodeulComplex = (cx, cz, h) =>
  cx > NODEUL_COMPLEX.x0 && cx < NODEUL_COMPLEX.x1 && cz > NODEUL_COMPLEX.z0 && cz < NODEUL_COMPLEX.z1 && h > NODEUL_COMPLEX.minH;
// island extent (for the trees, lamps and the revetment treatment in the terrain shader)
export const NODEUL_ISLAND = { x0: 820, x1: 1500, z0: 630, z1: 1010 };   // also used in src/wgsl/terrain.js (onNodeul)

const GLASS = [0.30, 0.36, 0.40];
const WHITE = [0.80, 0.80, 0.77];
const LAWN = [0.15, 0.21, 0.08];
const PAVING = [0.52, 0.50, 0.46];
const RAIL = [0.70, 0.71, 0.72];

const ringArea = (r) => { let s = 0; for (let i = 0; i < r.length; i++) { const a = r[i], b = r[(i + 1) % r.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; };
function pointInRing(x, z, r) {
  let inside = false;
  for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
    const [xi, zi] = r[i], [xj, zj] = r[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}
// drop near-duplicate and collinear points (OSM outlines carry a few)
export function clean(r) {
  let p = r.filter((a, i) => { const b = r[(i + 1) % r.length]; return Math.hypot(a[0] - b[0], a[1] - b[1]) > 0.3; });
  p = p.filter((b, i) => {
    const a = p[(i - 1 + p.length) % p.length], c = p[(i + 1) % p.length];
    return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0.05;
  });
  return p;
}
// inward offset with mitred corners; null if the result is not a valid shrunken copy
export function insetRing(r, d) {
  const ccw = ringArea(r) > 0;
  const n = r.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = r[(i - 1 + n) % n], b = r[i], c = r[(i + 1) % n];
    const e1 = norm2([b[0] - a[0], b[1] - a[1]]), e2 = norm2([c[0] - b[0], c[1] - b[1]]);
    // inward normals (left of the edge for a ccw ring)
    const s = ccw ? 1 : -1;
    const n1 = [-e1[1] * s, e1[0] * s], n2 = [-e2[1] * s, e2[0] * s];
    const m = norm2([n1[0] + n2[0], n1[1] + n2[1]]);
    const k = Math.min(3, 1 / Math.max(0.3, m[0] * n1[0] + m[1] * n1[1]));
    out.push([b[0] + m[0] * d * k, b[1] + m[1] * d * k]);
  }
  const a0 = Math.abs(ringArea(r)), a1 = Math.abs(ringArea(out));
  if (!(a1 > 30 && a1 < a0 * 0.95) || (ringArea(out) > 0) !== ccw) return null;
  for (const p of out) if (!pointInRing(p[0], p[1], r)) return null;
  return out;
}
const norm2 = (v) => { const l = Math.hypot(v[0], v[1]) || 1; return [v[0] / l, v[1] / l]; };

function walls(mb, r, y0, y1) {
  const ccw = ringArea(r) > 0;
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length];
    let nx = b[1] - a[1], nz = -(b[0] - a[0]);
    if (!ccw) { nx = -nx; nz = -nz; }
    const v0 = mb.vertex(a[0], y0, a[1], nx, 0, nz), v1 = mb.vertex(b[0], y0, b[1], nx, 0, nz);
    const v2 = mb.vertex(b[0], y1, b[1], nx, 0, nz), v3 = mb.vertex(a[0], y1, a[1], nx, 0, nz);
    mb.tri(v0, v1, v2); mb.tri(v0, v2, v3);
  }
}
function cap(mb, r, y) {
  const tri = earcut(r.flat());
  const base = mb.vcount;
  for (const p of r) mb.vertex(p[0], y, p[1], 0, 1, 0);
  for (let i = 0; i < tri.length; i += 3) mb.tri(base + tri[i], base + tri[i + 2], base + tri[i + 1]);
}
function railRing(mb, r, y, lights, lite) {
  mb.material(MAT.STEEL, RAIL, 0.35, 0.8);
  for (let i = 0; i < r.length; i++) {
    const a = r[i], b = r[(i + 1) % r.length];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    mb.tube([a[0], y + 1.1, a[1]], [b[0], y + 1.1, b[1]], 0.04, lite ? 4 : 6);
    if (!lite) mb.tube([a[0], y + 0.55, a[1]], [b[0], y + 0.55, b[1]], 0.015, 4);
    const nPost = Math.max(1, Math.round(L / 2.5));
    for (let k = 0; k < nPost; k++) {
      const t = k / nPost, x = a[0] + (b[0] - a[0]) * t, z = a[1] + (b[1] - a[1]) * t;
      mb.tube([x, y, z], [x, y + 1.1, z], 0.025, 4);
    }
    // low warm terrace lights along the edge (seen across the water at night)
    for (let s = 6; s < L - 3; s += 14) {
      const t = s / L;
      lights.push(a[0] + (b[0] - a[0]) * t, y + 0.9, a[1] + (b[1] - a[1]) * t, 0.12, 1.0, 0.8, 0.55, -10);
    }
  }
}

// The complex: one stepped building per footprint. base = ground under it, h = OSM-derived height.
function complex(mb, list, lights, lite) {
  let k = 0;
  for (const b of list) {
    let r = clean(b.ring);
    if (r.length < 3) continue;
    const seed = 900 + k++;
    // storeys: glazed ground floor (5 m: the hall / live house foyer), then white upper storeys
    // set back 3 m per step from the river-facing edges (estimated from photographs)
    const levels = [];
    const H = Math.max(b.h, 6);
    // the largest footprint (the live house / hall) has the glazed ground floor
    const big = Math.abs(ringArea(r)) > 900;
    levels.push({ ring: r, y0: b.base, y1: b.base + Math.min(5, H), glass: big });
    let cur = r, y = b.base + 5;
    while (y < b.base + H - 1.5) {
      const nxt = insetRing(cur, 3.2);
      if (!nxt) break;
      const y1 = Math.min(b.base + H, y + 3.8);
      levels.push({ ring: nxt, y0: y, y1, glass: false });
      cur = nxt; y = y1;
    }
    for (const lv of levels) {
      mb.setAux(lv.y0, lv.glass ? 5 : 3.8);
      if (lv.glass) mb.material(MAT.CURTAIN, GLASS, 0.1, 0, seed);
      else mb.material(MAT.OFFICE, WHITE, 0.8, 0, seed);
      walls(mb, lv.ring, lv.y0, lv.y1);
      mb.setAux(0, 0);
      // roof: a white slab edge, then lawn (the roofs are planted and walkable)
      mb.material(MAT.CONCRETE, WHITE, 0.85);
      walls(mb, lv.ring, lv.y1, lv.y1 + 0.35);
      mb.material(MAT.PBR, LAWN, 0.95);
      cap(mb, lv.ring, lv.y1 + 0.3);
      const inner = insetRing(lv.ring, 0.4);
      railRing(mb, inner || lv.ring, lv.y1 + 0.3, lights, lite);
    }
  }
}

// river distance (metres, > 0 on land) from the committed SDF grid
function sdfSampler(meta, sdf) {
  const s = meta.sdf;
  const grid = new Int16Array(sdf.buffer, sdf.byteOffset, sdf.byteLength / 2);
  return (x, z) => {
    const fx = (x - s.x0) / s.cell, fz = (z - s.z0) / s.cell;
    const i = Math.floor(fx), j = Math.floor(fz), tx = fx - i, tz = fz - j;
    if (i < 0 || j < 0 || i >= s.w - 1 || j >= s.h - 1) return -100;
    const g = (a, b) => grid[b * s.w + a] * s.scale;
    return (g(i, j) * (1 - tx) + g(i + 1, j) * tx) * (1 - tz) + (g(i, j + 1) * (1 - tx) + g(i + 1, j + 1) * tx) * tz;
  };
}
// ground height near the water (mirrors bankProfile in src/wgsl/terrain.js for 5 < d < 40)
const promenadeY = (d) => d < 14 ? 2.6 + (d - 5) * 0.01 : 2.7 + ((Math.min(d, 40) - 14) / 26) ** 2 * 1.3;
// island lawn west of the complex (노들섬 잔디마당): an ellipse along the island axis; centre and
// half-axes estimated from aerial photographs
const LAWN_C = [1035, 768], LAWN_AX = norm2([573, 328]), LAWN_RA = 92, LAWN_RB = 40;
function inLawn(x, z) {
  const dx = x - LAWN_C[0], dz = z - LAWN_C[1];
  const a = dx * LAWN_AX[0] + dz * LAWN_AX[1], b = -dx * LAWN_AX[1] + dz * LAWN_AX[0];
  return (a / LAWN_RA) ** 2 + (b / LAWN_RB) ** 2 < 1;
}

// Closed line along the island at distance d from the water: rays from the island's centre line
function islandContour(sdfAt, d) {
  // the island is long (WNW-ESE): march perpendicular to its axis from points on the axis, and
  // round the two tips with fans
  const A = [905, 662], B = [1478, 990];
  const L = Math.hypot(B[0] - A[0], B[1] - A[1]), u = [(B[0] - A[0]) / L, (B[1] - A[1]) / L], n = [-u[1], u[0]];
  const march = (o, dir) => {
    let last = null;
    for (let t = 0; t < 200; t += 1) {
      const x = o[0] + dir[0] * t, z = o[1] + dir[1] * t;
      if (sdfAt(x, z) < d) return last;
      last = [x, z];
    }
    return last;
  };
  const left = [], right = [];
  for (let s = 40; s <= L - 40; s += 6) {
    const o = [A[0] + u[0] * s, A[1] + u[1] * s];
    if (sdfAt(o[0], o[1]) < d) continue;
    const pL = march(o, n), pR = march(o, [-n[0], -n[1]]);
    if (pL) left.push(pL);
    if (pR) right.push(pR);
  }
  const tip = (s, sign) => {
    const o = [A[0] + u[0] * s, A[1] + u[1] * s];
    const out = [];
    for (let a = -80; a <= 80; a += 10) {
      const c = Math.cos(a * Math.PI / 180), sn = Math.sin(a * Math.PI / 180);
      const dir = [sign * (u[0] * c) + n[0] * sn, sign * (u[1] * c) + n[1] * sn];
      const p = march(o, dir);
      if (p) out.push(p);
    }
    return out;
  };
  return [...left, ...tip(L - 40, 1).reverse(), ...right.reverse(), ...tip(40, -1)];
}
function resample(poly, step) {
  const out = [];
  let acc = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    let s = step - acc;
    while (s <= L) { const t = s / L; out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]); s += step; }
    acc = (acc + L) % step;
  }
  return out;
}
const distToSeg = (p, a, b) => {
  const dx = b[0] - a[0], dz = b[1] - a[1], L2 = dx * dx + dz * dz;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / L2));
  return Math.hypot(p[0] - a[0] - dx * t, p[1] - a[1] - dz * t);
};

export function buildNodeul({ meta, sdf, complexList = [], lite = false }) {
  const mb = new MeshBuilder(1 << 13);
  const lights = [];
  complex(mb, complexList, lights, lite);

  const sdfAt = sdfSampler(meta, sdf);
  const bridges = meta.bridges.filter((b) => b.type === 'hangang');
  // the road across the island between the two bridge sections
  const h0 = bridges.find((b) => b.key === 'hangang0'), h1 = bridges.find((b) => b.key === 'hangang1');
  const road = h0 && h1 ? [h0.b, h1.a] : null;
  const nearRoad = (p, w) => bridges.some((b) => distToSeg(p, b.a, b.b) < w) || (road && distToSeg(p, road[0], road[1]) < w);

  // ---- promenade lamps: 4 m posts every ~28 m, 9 m in from the water's edge ----
  const contour = islandContour(sdfAt, 9);
  const lampPts = resample(contour, 28).filter((p) => !nearRoad(p, 16));
  for (const [x, z] of lampPts) {
    const y = promenadeY(sdfAt(x, z));
    mb.material(MAT.STEEL, [0.22, 0.23, 0.24], 0.5, 0.6);
    mb.tube([x, y - 0.3, z], [x, y + 3.9, z], 0.07, lite ? 4 : 6);
    mb.material(MAT.LAMP, [1.0, 0.82, 0.6], 0.5, 0, 0, 0.5);
    mb.box([x, y + 4.0, z], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.16, 0.12, 0.16);
    lights.push(x, y + 4.0, z, 0.2, 1.0, 0.8, 0.55, -10);
  }

  // ---- abutments: where each bridge section meets the island, a wall under the deck end and
  // splayed wing walls along the bank, faced in grey granite-coloured concrete ----
  for (const br of bridges) {
    const fr = bridgeFrame(br);
    // station of the island-side end of the river spans
    const onIslandAtEnd = Math.hypot(br.b[0] - 1190, br.b[1] - 830) < Math.hypot(br.a[0] - 1190, br.a[1] - 830);
    const s = onIslandAtEnd ? br.water[1] + 4 : br.water[0] - 4;
    const dir = onIslandAtEnd ? 1 : -1;     // towards the island along the axis
    const yTop = fr.deck(s) - 0.4;
    const half = br.deckW / 2 + 1.5;
    mb.material(MAT.CONCRETE, [0.52, 0.51, 0.48], 0.9);
    const c = fr.P(s, 0, 0);
    mb.box([c[0], (yTop - 1.5) / 2, c[2]], fr.n, [0, 1, 0], fr.u, half, (yTop + 1.5) / 2, 1.0);
    for (const side of [-1, 1]) {
      const a = fr.P(s, side * half, 0), b = fr.P(s + dir * 10, side * (half + 7), 0);
      const mid = [(a[0] + b[0]) / 2, (a[2] + b[2]) / 2];
      const L = Math.hypot(b[0] - a[0], b[2] - a[2]);
      const w = [(b[0] - a[0]) / L, 0, (b[2] - a[2]) / L], uu = [w[2], 0, -w[0]];
      const hh = Math.max(1.5, yTop - 0.5);
      mb.box([mid[0], (hh - 1.5) / 2, mid[1]], uu, [0, 1, 0], w, 0.45, (hh + 1.5) / 2, L / 2);
    }
  }
  // ---- a row of trees 4 m inland of the lamps, every ~11 m, on the western half (the east end
  // is already woodland in the data) ----
  const treeRow = [];
  let k = 0;
  for (const [x, z] of resample(islandContour(sdfAt, 13), 11)) {
    if (x > 1150 || x < 1000 || inLawn(x, z) || nearRoad([x, z], 18)) continue;
    const r = (Math.imul(++k, 2654435761) >>> 0) / 4294967296;
    treeRow.push(x, z, promenadeY(sdfAt(x, z)), 0.85 + 0.35 * r, r);
  }
  return { mesh: mb.result(), lights: new Float32Array(lights), treeRow: new Float32Array(treeRow) };
}

// Tree instances (x, z, ground y, scale, random) with the lawn cleared and the promenade row added.
export function editNodeulTrees(trees, treeRow) {
  const t = new Float32Array(trees.buffer, trees.byteOffset, trees.byteLength / 4);
  const out = new Float32Array(t.length + treeRow.length);
  let o = 0;
  for (let i = 0; i < t.length; i += 5) {
    if (inLawn(t[i], t[i + 1])) continue;
    out.set(t.subarray(i, i + 5), o); o += 5;
  }
  out.set(treeRow, o); o += treeRow.length;
  return new Uint8Array(out.buffer, 0, o * 4);
}
