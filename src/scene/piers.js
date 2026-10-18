// Piers from OSM: the Yeouido E-Land Cruise terminal (a floating terminal building) and the
// smaller pontoons nearby. Floating structures sit on pontoons ~0.9 m above the water.
import earcut from 'earcut';
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

function extrude(mb, pts, y0, y1, roofMat, wallMat, wallCol, roofCol) {
  const n = pts.length;
  let area = 0;
  for (let i = 0; i < n; i++) { const a = pts[i], b = pts[(i + 1) % n]; area += a[0] * b[1] - b[0] * a[1]; }
  const ccw = area > 0;
  mb.material(wallMat, wallCol, 0.6);
  for (let i = 0; i < n; i++) {
    const a = pts[i], b = pts[(i + 1) % n];
    let nx = b[1] - a[1], nz = -(b[0] - a[0]);
    if (!ccw) { nx = -nx; nz = -nz; }
    const v0 = mb.vertex(a[0], y0, a[1], nx, 0, nz), v1 = mb.vertex(b[0], y0, b[1], nx, 0, nz);
    const v2 = mb.vertex(b[0], y1, b[1], nx, 0, nz), v3 = mb.vertex(a[0], y1, a[1], nx, 0, nz);
    mb.tri(v0, v1, v2); mb.tri(v0, v2, v3);
  }
  const flat = pts.flat();
  const tri = earcut(flat);
  mb.material(roofMat, roofCol, 0.8);
  const base = mb.vcount;
  for (const p of pts) mb.vertex(p[0], y1, p[1], 0, 1, 0);
  for (let i = 0; i < tri.length; i += 3) mb.tri(base + tri[i], base + tri[i + 2], base + tri[i + 1]);
}

function inset(pts, d) {
  // shrink a polygon towards its centroid (good enough for the convex-ish terminal outline)
  const c = pts.reduce((s, p) => [s[0] + p[0], s[1] + p[1]], [0, 0]).map((v) => v / pts.length);
  return pts.map(([x, z]) => { const dx = x - c[0], dz = z - c[1], l = Math.hypot(dx, dz) || 1; return [x - dx / l * d, z - dz / l * d]; });
}

export function buildPiers(piers) {
  const mb = new MeshBuilder(4096);
  for (const p of piers) {
    let pts = p.pts.slice();
    if (pts.length > 2 && pts[0][0] === pts[pts.length - 1][0] && pts[0][1] === pts[pts.length - 1][1]) pts.pop();
    if (pts.length < 3) continue;
    // pontoon hull (dark steel) + deck
    extrude(mb, pts, -0.6, 0.9, MAT.DECK, MAT.STEEL, [0.2, 0.22, 0.24], [0.42, 0.42, 0.4]);
    if (p.building) {
      // Two-storey floating terminal: glazed volumes between white floor slabs, the upper floor
      // set back behind a roof terrace with a railing (massing approximated from photographs).
      const white = [0.82, 0.82, 0.8];
      mb.setAux(0.9, 3.6);
      extrude(mb, inset(pts, 3.0), 0.9, 4.4, MAT.ROOF, MAT.CURTAIN, [0.34, 0.42, 0.48], white);
      mb.setAux(0, 0);
      extrude(mb, inset(pts, 2.2), 4.4, 5.0, MAT.CONCRETE, MAT.CONCRETE, white, white);
      mb.setAux(5.0, 3.2);
      extrude(mb, inset(pts, 5.5), 5.0, 8.2, MAT.ROOF, MAT.CURTAIN, [0.30, 0.38, 0.44], white);
      mb.setAux(0, 0);
      extrude(mb, inset(pts, 4.6), 8.2, 8.7, MAT.CONCRETE, MAT.CONCRETE, white, [0.5, 0.5, 0.5]);
      // terrace railing on the lower slab, and railing posts around the pontoon edge
      mb.material(MAT.STEEL, [0.7, 0.72, 0.74], 0.3, 0.9);
      const t = inset(pts, 2.4);
      for (let i = 0; i < t.length; i++) {
        const a = t[i], c = t[(i + 1) % t.length];
        mb.tube([a[0], 6.05, a[1]], [c[0], 6.05, c[1]], 0.04, 5);
      }
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], c = pts[(i + 1) % pts.length];
        mb.tube([a[0], 1.95, a[1]], [c[0], 1.95, c[1]], 0.03, 5);
      }
    }
  }
  return mb.result();
}
