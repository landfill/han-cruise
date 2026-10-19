// The cruise boat, modelled at real scale in its own frame: -z forward (bow), +x starboard,
// y = 0 at the waterline. A ~50 m two-deck river cruiser with an open observation deck.
// Dimensions follow the class of excursion boats used on the Han (about 50 m x 11 m,
// 1.5 m draft); this is a generic vessel, not a replica of a specific ship.
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

export const BOAT = {
  length: 50, beam: 11, draft: 1.5,
  deck1: 1.4, deck2: 3.8, top: 6.0,
  eye: 1.62,
};

const WHITE = [0.86, 0.87, 0.86];
const NAVY = [0.07, 0.12, 0.22];
const DECK_GREEN = [0.22, 0.27, 0.24];
const RAIL = [0.62, 0.63, 0.64];
const WOOD = [0.42, 0.27, 0.16];
const TEAK = [0.40, 0.28, 0.18];
const RUBBER = [0.035, 0.035, 0.037];
const FITTING = [0.13, 0.14, 0.15];   // black-painted deck fittings
const ROPE = [0.70, 0.64, 0.50];

// half-beam of the hull at a station (z) and height (y)
function halfBeam(z, y) {
  const L = BOAT.length, B = BOAT.beam / 2;
  const t = (z + L / 2) / L; // 0 at bow, 1 at stern
  let f;
  if (t < 0.32) { const u = t / 0.32; f = Math.sin(u * Math.PI / 2) ** 0.85; } // fine entry
  else if (t > 0.94) f = 1 - (t - 0.94) * 1.2;  // slight tuck at the transom
  else f = 1;
  // flare above the waterline, deadrise below
  const flare = y > 0 ? 1 + y * 0.012 : 1 + y * 0.09;
  return Math.max(0.05, B * f * flare);
}
function sheer(z) {
  const t = (z + BOAT.length / 2) / BOAT.length;
  return BOAT.deck1 + 0.55 * Math.max(0, 1 - t / 0.35) ** 2;
}

function hull(mb) {
  const L = BOAT.length;
  const N = 40;
  const ys = (z) => [-BOAT.draft, -BOAT.draft * 0.75, -0.6, 0, 0.5, sheer(z)];
  const rows = [];
  for (let i = 0; i <= N; i++) {
    const z = -L / 2 + L * i / N;
    const t = i / N;
    const keel = t < 0.06 ? -BOAT.draft + (0.06 - t) / 0.06 * BOAT.draft * 0.8 : -BOAT.draft;
    const pts = ys(z).map((y, k) => {
      const yy = k === 0 ? keel : y;
      const w = k === 0 ? 0.25 : halfBeam(z, yy);
      return [w, yy, z];
    });
    rows.push(pts);
  }
  for (const side of [-1, 1]) {
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < rows[i].length - 1; k++) {
        const a = rows[i][k], b = rows[i][k + 1], c = rows[i + 1][k + 1], d = rows[i + 1][k];
        const P = (p) => [p[0] * side, p[1], p[2]];
        // paint: antifouling below water, navy boot-top band, white topsides
        const ym = (a[1] + b[1]) / 2;
        if (ym < -0.15) mb.material(MAT.HULL, [0.35, 0.09, 0.07], 0.5);
        else if (ym < 0.45) mb.material(MAT.HULL, NAVY, 0.35);
        else mb.material(MAT.HULL, WHITE, 0.3);
        if (side > 0) mb.quad(P(a), P(d), P(c), P(b)); else mb.quad(P(a), P(b), P(c), P(d));
      }
    }
  }
  // transom
  mb.material(MAT.HULL, WHITE, 0.3);
  const st = rows[N];
  for (let k = 0; k < st.length - 1; k++) {
    const a = st[k], b = st[k + 1];
    mb.quad([-a[0], a[1], a[2]], [a[0], a[1], a[2]], [b[0], b[1], b[2]], [-b[0], b[1], b[2]]);
  }
  // main deck (fore deck and side decks)
  mb.material(MAT.DECK, DECK_GREEN, 0.7);
  for (let i = 0; i < N; i++) {
    const a = rows[i][rows[i].length - 1], b = rows[i + 1][rows[i + 1].length - 1];
    mb.quad([-a[0], a[1], a[2]], [-b[0], b[1], b[2]], [b[0], b[1], b[2]], [a[0], a[1], a[2]]);
  }
  // bulwark rail around the main deck
  mb.material(MAT.STEEL, RAIL, 0.25, 0.9);
  for (let i = 0; i < N; i++) {
    const a = rows[i][rows[i].length - 1], b = rows[i + 1][rows[i + 1].length - 1];
    for (const side of [-1, 1]) {
      const pa = [a[0] * side - side * 0.1, a[1] + 1.0, a[2]], pb = [b[0] * side - side * 0.1, b[1] + 1.0, b[2]];
      mb.tube(pa, pb, 0.03, 5);
      if (i % 2 === 0) mb.tube([a[0] * side - side * 0.1, a[1], a[2]], pa, 0.025, 4);
    }
  }
  return rows;
}

const polyTube = (mb, pts, r, sides) => { for (let i = 0; i < pts.length - 1; i++) mb.tube(pts[i], pts[i + 1], r, sides); };

// Rubbing strakes (black rubber D-fenders that take the knocks at the piers) along the sheer and
// at the knuckle above the boot-top, round the transom too, and a thin LED accent line under the
// upper strake that the Han cruisers light at night.
function hullFittings(mb, rows, lite) {
  const N = rows.length - 1;
  const top = (r, side, out) => { const p = r[r.length - 1]; return [side * (p[0] + out), p[1] - 0.16, p[2]]; };
  const low = (r, side, out) => { const p = r[4]; return [side * (p[0] + out), p[1], p[2]]; };
  mb.material(MAT.RUBBER, RUBBER, 0.7);
  for (const side of [-1, 1]) {
    polyTube(mb, rows.map((r) => top(r, side, 0.05)), 0.11, lite ? 5 : 8);
    polyTube(mb, rows.slice(2).map((r) => low(r, side, 0.04)), 0.075, lite ? 4 : 6);
  }
  // across the transom
  const st = rows[N];
  mb.tube(top(st, -1, 0.0), top(st, 1, 0.0), 0.11, lite ? 5 : 8);
  mb.tube(low(st, -1, 0.0), low(st, 1, 0.0), 0.075, lite ? 4 : 6);
  // LED accent line (emissive at night, a pale channel by day)
  mb.material(MAT.LAMP, [0.55, 0.78, 1.0], 0.4, 0, 0, 0.14);
  for (const side of [-1, 1]) {
    for (let i = 3; i < N - 1; i++) {
      const a = rows[i][rows[i].length - 1], b = rows[i + 1][rows[i + 1].length - 1];
      const ya = a[1] - 0.42, yb = b[1] - 0.42;
      const xa = side * (halfBeam(a[2], ya) + 0.012), xb = side * (halfBeam(b[2], yb) + 0.012);
      mb.quad([xa, ya - 0.025, a[2]], [xb, yb - 0.025, b[2]], [xb, yb + 0.025, b[2]], [xa, ya + 0.025, a[2]]);
    }
  }
}

// Double bitt (mooring bollard pair) on a base plate, axis along the boat (yaw 0) or turned.
function bitt(mb, x, y, z, yaw, lite) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  mb.material(MAT.STEEL, FITTING, 0.45, 0.6);
  mb.box([x, y + 0.025, z], [c, 0, -s], [0, 1, 0], [s, 0, c], 0.13, 0.025, 0.32);
  for (const o of [-0.17, 0.17]) {
    mb.lathe(x + s * o, y + 0.05, z + c * o, [[0.085, 0], [0.08, 0.26], [0.115, 0.28], [0.115, 0.33], [0.06, 0.345], [0.0, 0.35]], lite ? 8 : 12);
  }
}
// Mooring line flaked down in a coil next to the bitts.
function ropeCoil(mb, x, y, z, R, lite) {
  mb.material(MAT.PBR, ROPE, 0.95);
  const turns = lite ? 2 : 4;
  for (let k = 0; k < turns; k++) {
    const r = R - k * 0.045, cy = y + 0.022 + (k % 2) * 0.012;
    mb.lathe(x, cy, z, [[r - 0.022, 0], [r, -0.02], [r + 0.022, 0], [r, 0.02], [r - 0.022, 0]], lite ? 12 : 20);
  }
}
// Fairlead / panama chock on the bulwark: a short steel block with a dark opening
function chock(mb, x, y, z, side) {
  mb.material(MAT.STEEL, FITTING, 0.45, 0.6);
  mb.box([x, y + 0.12, z], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.07, 0.12, 0.2);
  mb.material(MAT.PBR, [0.02, 0.02, 0.02], 0.9);
  mb.box([x + side * 0.072, y + 0.13, z], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.004, 0.06, 0.11);
}

// Inflatable liferaft canister (white GRP capsule with a seam band) on a steel cradle
function liferaft(mb, x, y, z, lite) {
  mb.material(MAT.STEEL, [0.55, 0.56, 0.57], 0.4, 0.8);
  for (const dx of [-0.42, 0.42]) mb.box([x + dx, y + 0.12, z], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.03, 0.12, 0.3);
  mb.box([x, y + 0.015, z], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.6, 0.015, 0.33);
  // lashing straps: one dark band texture would need UVs; three short dark rings are cheap
  mb.material(MAT.PBR, [0.12, 0.12, 0.13], 0.8);
  for (const dx of [-0.4, 0.4]) mb.tube([x + dx - 0.03, y + 0.52, z], [x + dx + 0.03, y + 0.52, z], 0.285, lite ? 10 : 16);
  mb.material(MAT.HULL, [0.9, 0.9, 0.87], 0.35);
  const cy = y + 0.52;
  mb.tube([x - 0.62, cy, z], [x + 0.62, cy, z], 0.28, lite ? 10 : 16, true);
  if (!lite) {
    // end domes
    for (const e of [-1, 1]) mb.tube([x + e * 0.62, cy, z], [x + e * 0.7, cy, z], 0.22, 14, true);
  }
}

function cabin(mb, z0, z1, y0, y1, inset, windowBand, frontRake = 0, roofMat = MAT.DECK, roofCol = DECK_GREEN) {
  // walls with a continuous window band
  const hw = (z) => halfBeam(z, y0) - inset;
  const N = 16;
  const zs = [];
  for (let i = 0; i <= N; i++) zs.push(z0 + (z1 - z0) * i / N);
  const wy0 = y0 + windowBand[0], wy1 = y0 + windowBand[1];
  for (const side of [-1, 1]) {
    for (let i = 0; i < N; i++) {
      const za = zs[i], zb = zs[i + 1];
      const xa = hw(za) * side, xb = hw(zb) * side;
      const Q = (x0, x1, ya, yb) => side > 0 ? mb.quad([x0, ya, za], [x1, ya, zb], [x1, yb, zb], [x0, yb, za]) : mb.quad([x1, ya, zb], [x0, ya, za], [x0, yb, za], [x1, yb, zb]);
      mb.material(MAT.HULL, WHITE, 0.35); Q(xa, xb, y0, wy0);
      mb.material(MAT.CABIN_GLASS, [0.05, 0.06, 0.07], 0.05, 0, 1); Q(xa, xb, wy0, wy1);
      mb.material(MAT.HULL, WHITE, 0.35); Q(xa, xb, wy1, y1);
    }
  }
  // front (raked) and back walls
  for (const [z, dir] of [[z0, -1], [z1, 1]]) {
    const w = hw(z);
    const rake = dir < 0 ? frontRake : 0;
    mb.material(MAT.HULL, WHITE, 0.35);
    mb.quad([-w, y0, z], [w, y0, z], [w, wy0, z], [-w, wy0, z]);
    mb.material(MAT.CABIN_GLASS, [0.05, 0.06, 0.07], 0.05, 0, 1);
    mb.quad([-w, wy0, z], [w, wy0, z], [w, wy1, z + rake * 0.6], [-w, wy1, z + rake * 0.6]);
    mb.material(MAT.HULL, WHITE, 0.35);
    mb.quad([-w, wy1, z + rake * 0.6], [w, wy1, z + rake * 0.6], [w, y1, z + rake], [-w, y1, z + rake]);
  }
  // roof (= next deck), slight overhang
  const hz0 = z0 + frontRake, hz1 = z1;
  const rw0 = hw(hz0) + 0.35, rw1 = hw(hz1) + 0.35;
  mb.material(roofMat, roofCol, 0.7);
  mb.quad([-rw0, y1, hz0 - 0.3], [-rw1, y1, hz1 + 0.3], [rw1, y1, hz1 + 0.3], [rw0, y1, hz0 - 0.3]);
  mb.material(MAT.HULL, WHITE, 0.35);
  mb.quad([-rw0, y1 - 0.25, hz0 - 0.3], [rw0, y1 - 0.25, hz0 - 0.3], [rw0, y1, hz0 - 0.3], [-rw0, y1, hz0 - 0.3]);
  mb.quad([rw1, y1 - 0.25, hz1 + 0.3], [-rw1, y1 - 0.25, hz1 + 0.3], [-rw1, y1, hz1 + 0.3], [rw1, y1, hz1 + 0.3]);
  for (const s of [-1, 1]) {
    const a = [s * rw0, y1 - 0.25, hz0 - 0.3], b = [s * rw1, y1 - 0.25, hz1 + 0.3];
    if (s > 0) mb.quad(a, b, [b[0], y1, b[2]], [a[0], y1, a[2]]); else mb.quad(b, a, [a[0], y1, a[2]], [b[0], y1, b[2]]);
  }
  // cabin interior lights at night (seen through the windows)
  return { z0: hz0 - 0.3, z1: hz1 + 0.3, w0: rw0, w1: rw1 };
}

function railing(mb, pts, y, height = 1.1, lite = false) {
  // stainless posts every ~1.4 m, top handrail (wood), three wire rails
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, az] = pts[i], [bx, bz] = pts[i + 1];
    const L = Math.hypot(bx - ax, bz - az);
    const n = Math.max(1, Math.round(L / 1.4));
    mb.material(MAT.STEEL, RAIL, 0.32, 0.6);
    for (let k = 0; k < n; k++) {
      const t = k / n;
      const x = ax + (bx - ax) * t, z = az + (bz - az) * t;
      mb.tube([x, y, z], [x, y + height, z], 0.025, lite ? 4 : 6);
    }
    // three wire rails (sub-pixel on a phone, where one heavier rail stands in for them)
    if (lite) mb.tube([ax, y + 0.5, az], [bx, y + 0.5, bz], 0.008, 3);
    else for (const h of [0.25, 0.5, 0.75]) mb.tube([ax, y + h, az], [bx, y + h, bz], 0.006, 4);
    // brushed stainless handrail tube
    mb.tube([ax, y + height + 0.02, az], [bx, y + height + 0.02, bz], 0.024, 8);
  }
}

function bench(mb, x, z, len, yaw, y) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const R = (lx, lz) => [x + c * lx - s * lz, z + s * lx + c * lz];
  mb.material(MAT.STEEL, [0.3, 0.31, 0.32], 0.4, 0.8);
  for (const lx of [-len / 2 + 0.2, len / 2 - 0.2]) {
    const [px, pz] = R(lx, 0);
    mb.box([px, y + 0.22, pz], [c, 0, s], [0, 1, 0], [-s, 0, c], 0.03, 0.22, 0.22);
  }
  mb.material(MAT.PBR, WOOD, 0.55);
  for (const lz of [-0.15, 0, 0.15]) {
    const [px, pz] = R(0, lz);
    mb.box([px, y + 0.45, pz], [c, 0, s], [0, 1, 0], [-s, 0, c], len / 2, 0.02, 0.06);
  }
  // backrest
  for (const lh of [0.62, 0.78]) {
    const [px, pz] = R(0, 0.24);
    mb.box([px, y + lh, pz], [c, 0, s], [0, 1, 0], [-s, 0, c], len / 2, 0.05, 0.02);
  }
}

// lite: the phone version leaves out sub-centimetre parts (wire rails, flanges, rope turns) and
// uses fewer sides on round parts; the outline, fittings and lights stay the same.
export function buildBoat({ lite = false } = {}) {
  const mb = new MeshBuilder(1 << 14);
  const rows = hull(mb);
  hullFittings(mb, rows, lite);
  const L = BOAT.length;
  // first cabin (main deck saloon), second cabin with the wheelhouse at its front; the roof of
  // the second is the open observation deck, laid with teak-look planks
  const mid = cabin(mb, -L / 2 + 9, L / 2 - 4, BOAT.deck1, BOAT.deck2, 0.9, [0.55, 2.05], 1.2);
  const top = cabin(mb, -L / 2 + 12.5, L / 2 - 7, BOAT.deck2, BOAT.top, 1.2, [0.5, 1.75], 0.9, MAT.DECK_WOOD, TEAK);
  // observation deck railing (follows the roof outline)
  const y = BOAT.top;
  const rail = [[-top.w0 + 0.15, top.z0 + 0.2], [top.w0 - 0.15, top.z0 + 0.2], [top.w1 - 0.15, top.z1 - 0.2], [-top.w1 + 0.15, top.z1 - 0.2], [-top.w0 + 0.15, top.z0 + 0.2]];
  railing(mb, rail, y, 1.1, lite);
  // benches in rows facing outboard, plus a few facing forward
  for (let z = top.z0 + 4; z < top.z1 - 3; z += 3.2) {
    bench(mb, -1.6, z, 2.4, Math.PI / 2, y);
    bench(mb, 1.6, z, 2.4, -Math.PI / 2, y);
  }
  // mast with navigation lights near the front of the observation deck, stack aft
  mb.material(MAT.STEEL, RAIL, 0.3, 0.9);
  const mz = top.z0 + 1.6;
  // short mast: river cruisers keep their air draft low for the bridges (about 8.5 m here)
  mb.tube([0, y, mz], [0, y + 2.4, mz], 0.07, 8);
  mb.tube([0, y + 1.9, mz], [0.7, y + 1.9, mz], 0.03, 4);
  mb.tube([0, y + 1.9, mz], [-0.7, y + 1.9, mz], 0.03, 4);
  // light housings (masthead on top, all-round light on the yard), radar scanner on a short
  // pedestal, whip antennas and a horn; all below the 8.4 m air draft of the mast
  mb.material(MAT.STEEL, FITTING, 0.4, 0.4);
  mb.box([0, y + 2.32, mz - 0.05], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.07, 0.07, 0.07);
  mb.lathe(0.62, y + 1.93, mz, [[0.05, 0], [0.05, 0.14], [0.0, 0.15]], 8);
  mb.lathe(0, y + 1.42, mz, [[0.1, 0], [0.09, 0.14], [0.0, 0.15]], 10);
  mb.material(MAT.HULL, [0.92, 0.92, 0.9], 0.35);
  mb.box([0, y + 1.62, mz], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.9, 0.05, 0.09);
  mb.material(MAT.STEEL, [0.85, 0.85, 0.85], 0.4, 0.2);
  for (const x of [-0.62, 0.45]) mb.tube([x, y + 1.9, mz], [x, y + 2.38, mz], 0.008, 3);
  mb.material(MAT.STEEL, [0.75, 0.75, 0.74], 0.35, 0.8);
  mb.lathe(-0.35, y + 1.95, mz, [[0.03, 0], [0.04, 0.05], [0.09, 0.17], [0.0, 0.17]], 10);
  mb.material(MAT.HULL, WHITE, 0.35);
  mb.box([0, y + 0.9, top.z1 - 2.2], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.9, 0.9, 0.7);
  mb.material(MAT.STEEL, [0.15, 0.15, 0.16], 0.5, 0.5);
  mb.tube([0, y + 1.8, top.z1 - 2.2], [0, y + 2.4, top.z1 - 2.2], 0.32, 10);
  // life rings on the railing
  mb.material(MAT.PBR, [0.85, 0.3, 0.05], 0.6);
  for (const [x, z] of [[-top.w0 + 0.2, top.z0 + 6], [top.w0 - 0.2, top.z0 + 6], [-top.w1 + 0.2, top.z1 - 5], [top.w1 - 0.2, top.z1 - 5]]) {
    mb.lathe(x, y + 0.55, z, [[0.3, -0.06], [0.36, 0], [0.3, 0.06], [0.24, 0], [0.3, -0.06]], 12, [0.25, 1]);
  }
  // stairs well (hatch) at the back of the observation deck
  mb.material(MAT.HULL, WHITE, 0.35);
  mb.box([2.6, y + 1.1, top.z1 - 5.5], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.9, 1.1, 1.4);
  // ---- mooring and ground tackle on the open main decks ----
  const fz = -L / 2 + 5.0;                // foredeck station
  const az = L / 2 - 1.8;                 // aft deck station
  for (const side of [-1, 1]) {
    const yf = sheer(fz), ya = sheer(az);
    const xf = side * (halfBeam(fz, yf) - 0.75), xa = side * (halfBeam(az, ya) - 0.8);
    bitt(mb, xf, yf, fz, 0.25 * side, lite);
    bitt(mb, xa, ya, az, -0.2 * side, lite);
    ropeCoil(mb, xf - side * 0.75, yf, fz + 1.1, 0.3, lite);
    ropeCoil(mb, xa - side * 0.85, ya, az - 1.0, 0.3, lite);
    chock(mb, side * (halfBeam(fz - 1.2, yf) - 0.12), sheer(fz - 1.2), fz - 1.2, side);
    chock(mb, side * (halfBeam(az + 0.6, ya) - 0.12), ya, az + 0.6, side);
  }
  {
    // anchor windlass on the centreline with the chain running to the hawse pipe at the stem
    const wz = -L / 2 + 3.2, wy = sheer(wz);
    mb.material(MAT.STEEL, FITTING, 0.45, 0.6);
    mb.box([0, wy + 0.2, wz], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.32, 0.2, 0.26);
    mb.material(MAT.STEEL, [0.3, 0.12, 0.08], 0.55, 0.5);
    mb.tube([-0.45, wy + 0.32, wz], [0.45, wy + 0.32, wz], 0.16, lite ? 8 : 12, true);
    mb.material(MAT.STEEL, [0.08, 0.08, 0.08], 0.6, 0.8);
    const hz = -L / 2 + 1.2, hy = sheer(hz) + 0.04;
    mb.tube([0, wy + 0.3, wz - 0.1], [0, hy, hz], 0.035, 4);
    if (!lite) for (let k = 0; k < 8; k++) {
      const t = (k + 0.5) / 8, zz = wz - 0.1 + (hz - wz + 0.1) * t, yy = wy + 0.3 + (hy - wy - 0.3) * t;
      mb.box([0, yy, zz], k % 2 ? [1, 0, 0] : [0, 1, 0], k % 2 ? [0, 1, 0] : [1, 0, 0], [0, 0, 1], 0.012, 0.045, 0.07);
    }
  }
  // liferaft canisters in a rack on the open roof aft of the observation deck (deck 2)
  for (const x of [-2.4, -0.9, 0.9, 2.4]) liferaft(mb, x, BOAT.deck2, mid.z1 - 1.5, lite);
  // ensign staff at the stern with the flag furled round it
  {
    const sz = L / 2 - 0.6, sy = sheer(sz);
    mb.material(MAT.STEEL, [0.8, 0.8, 0.8], 0.35, 0.8);
    mb.tube([0, sy, sz], [0, sy + 2.3, sz], 0.025, 6);
    mb.material(MAT.PBR, [0.85, 0.85, 0.83], 0.8);
    mb.tube([0, sy + 1.55, sz + 0.03], [0, sy + 2.25, sz + 0.03], 0.05, 6);
  }
  // LED strips under the roof edges of both decks (warm white, lit at night)
  mb.material(MAT.LAMP, [1.0, 0.88, 0.7], 0.5, 0, 0, 0.16);
  for (const [cb, yy] of [[mid, BOAT.deck2 - 0.26], [top, BOAT.top - 0.26]]) {
    for (const s of [-1, 1]) {
      const a = [s * (cb.w0 - 0.02), yy, cb.z0 + 0.3], b = [s * (cb.w1 - 0.02), yy, cb.z1 - 0.3];
      mb.quad(a, b, [b[0] - s * 0.06, b[1], b[2]], [a[0] - s * 0.06, a[1], a[2]]);
    }
  }

  // lights: masthead (white), sidelights (port red / starboard green), stern (white), deck lamps
  const lights = [
    0, y + 2.45, mz, 0.12, 1.0, 0.95, 0.85, -10,
    -halfBeam(-L / 2 + 14, 2) - 0.1, BOAT.deck2 + 0.3, -L / 2 + 14, 0.1, 1.0, 0.08, 0.04, -10,
    halfBeam(-L / 2 + 14, 2) + 0.1, BOAT.deck2 + 0.3, -L / 2 + 14, 0.1, 0.05, 1.0, 0.3, -10,
    0, BOAT.deck1 + 1.2, L / 2 - 0.2, 0.1, 1.0, 0.95, 0.85, -10,
    // floodlights over the fore and aft mooring decks
    0, BOAT.deck2 - 0.3, mid.z0 - 0.2, 0.09, 1.0, 0.88, 0.7, -10,
    0, BOAT.deck2 - 0.3, mid.z1 + 0.2, 0.09, 1.0, 0.88, 0.7, -10,
  ];
  const deckLamps = [];
  for (let z = top.z0 + 3; z < top.z1 - 2; z += 4.5) for (const x of [-top.w0 + 0.3, top.w0 - 0.3]) deckLamps.push(x, y + 1.25, z);
  mb.material(MAT.LAMP, [1.0, 0.86, 0.66], 0.5, 0, 0, 0.25);
  for (let i = 0; i < deckLamps.length; i += 3) mb.box([deckLamps[i], deckLamps[i + 1], deckLamps[i + 2]], [1, 0, 0], [0, 1, 0], [0, 0, 1], 0.05, 0.05, 0.05);
  for (let i = 0; i < deckLamps.length; i += 3) lights.push(deckLamps[i], deckLamps[i + 1], deckLamps[i + 2], 0.08, 0.5, 0.42, 0.32, -10);

  // walkable area of the observation deck (camera constraint), in boat coordinates
  const walk = { y: y, z0: top.z0 + 0.6, z1: top.z1 - 0.6, w0: top.w0 - 0.6, w1: top.w1 - 0.6 };
  const viewpoints = [
    { key: 'bow', name: '선수 전망', pos: [0, y + BOAT.eye, top.z0 + 1.0], yaw: 0, pitch: -3 },
    { key: 'port', name: '좌현', pos: [-(top.w0 - 0.7), y + BOAT.eye, -2], yaw: -90, pitch: -4 },
    { key: 'starboard', name: '우현', pos: [top.w0 - 0.7, y + BOAT.eye, -2], yaw: 90, pitch: -4 },
    { key: 'stern', name: '선미 (항적)', pos: [0, y + BOAT.eye, top.z1 - 1.0], yaw: 180, pitch: -8 },
    { key: 'foredeck', name: '하갑판 선수', pos: [0, BOAT.deck1 + 0.55 + BOAT.eye, -L / 2 + 5.5], yaw: 0, pitch: -2 },
  ];
  return { mesh: mb.result(), lights: new Float32Array(lights), walk, viewpoints };
}
