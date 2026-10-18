// Tree template: a lumpy crown (subdivided icosahedron, displaced) on a short trunk.
// Vertex: pos f32x3 | normal f32x3 | part f32 (0 trunk, 1 crown)  -> stride 28
export const TREE_STRIDE = 28;

export function buildTreeTemplate() {
  const v = [], idx = [];
  const push = (p, n, part) => { v.push(p[0], p[1], p[2], n[0], n[1], n[2], part); return v.length / 7 - 1; };
  // trunk: 5-sided
  const tr = 0.22, th = 3.2;
  for (let k = 0; k < 5; k++) {
    const a0 = (k / 5) * Math.PI * 2, a1 = ((k + 1) / 5) * Math.PI * 2;
    const n0 = [Math.cos(a0), 0, Math.sin(a0)], n1 = [Math.cos(a1), 0, Math.sin(a1)];
    const i0 = push([n0[0] * tr, 0, n0[2] * tr], n0, 0), i1 = push([n1[0] * tr, 0, n1[2] * tr], n1, 0);
    const i2 = push([n1[0] * tr * 0.7, th, n1[2] * tr * 0.7], n1, 0), i3 = push([n0[0] * tr * 0.7, th, n0[2] * tr * 0.7], n0, 0);
    idx.push(i0, i1, i2, i0, i2, i3);
  }
  // crown: icosahedron subdivided once, displaced into lobes
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map((p) => { const l = Math.hypot(...p); return p.map((c) => c / l); });
  let faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let s = 0; s < 1; s++) {
    const cache = new Map();
    const mid = (a, b) => {
      const key = a < b ? a + '_' + b : b + '_' + a;
      if (cache.has(key)) return cache.get(key);
      const p = verts[a].map((c, i) => (c + verts[b][i]) / 2);
      const l = Math.hypot(...p);
      verts.push(p.map((c) => c / l));
      cache.set(key, verts.length - 1);
      return verts.length - 1;
    };
    const nf = [];
    for (const [a, b, c] of faces) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    faces = nf;
  }
  const R = 3.2, cy = th + 2.6;
  const disp = (p) => 1 + 0.16 * Math.sin(p[0] * 5.1 + 1.3) * Math.sin(p[1] * 4.3) * Math.sin(p[2] * 5.7 + 0.4) + 0.08 * Math.sin(p[0] * 11 + p[2] * 9);
  const base = v.length / 7;
  for (const p of verts) {
    const k = disp(p);
    push([p[0] * R * k, cy + p[1] * R * 0.85 * k, p[2] * R * k], p, 1);
  }
  for (const [a, b, c] of faces) idx.push(base + a, base + c, base + b);
  return { vertices: new Float32Array(v), indices: new Uint16Array(idx), icount: idx.length };
}
