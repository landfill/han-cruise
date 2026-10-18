// Growable vertex/index buffers in the shared mesh vertex format (see src/wgsl/mesh.js):
// pos f32x3 | normal snorm8x4 | colour unorm8x4 | material u8x4 (id, rough, metal, seed) | aux f32x2
export const STRIDE = 32;

export class MeshBuilder {
  constructor(capacity = 4096) {
    this.buf = new ArrayBuffer(capacity * STRIDE);
    this.f = new Float32Array(this.buf);
    this.b = new Uint8Array(this.buf);
    this.i8 = new Int8Array(this.buf);
    this.vcount = 0;
    this.idx = new Uint32Array(capacity * 2);
    this.icount = 0;
    this.min = [Infinity, Infinity, Infinity];
    this.max = [-Infinity, -Infinity, -Infinity];
    // current material state
    this.col = [200, 200, 200, 255];
    this.mat = [0, 200, 0, 0];
    this.aux = [0, 0];
  }
  material(id, rgb, rough = 0.8, metal = 0, seed = 0, alpha = 1) {
    this.col = [rgb[0] * 255, rgb[1] * 255, rgb[2] * 255, alpha * 255].map((v) => Math.max(0, Math.min(255, Math.round(v))));
    this.mat = [id, Math.round(rough * 255), Math.round(metal * 255), seed & 255];
    return this;
  }
  setAux(a, b) { this.aux = [a, b]; return this; }
  _growV(n) {
    if ((this.vcount + n) * STRIDE <= this.buf.byteLength) return;
    let cap = this.buf.byteLength / STRIDE;
    while ((this.vcount + n) > cap) cap *= 2;
    const nb = new ArrayBuffer(cap * STRIDE);
    new Uint8Array(nb).set(this.b.subarray(0, this.vcount * STRIDE));
    this.buf = nb; this.f = new Float32Array(nb); this.b = new Uint8Array(nb); this.i8 = new Int8Array(nb);
  }
  _growI(n) {
    if (this.icount + n <= this.idx.length) return;
    let cap = this.idx.length;
    while (this.icount + n > cap) cap *= 2;
    const ni = new Uint32Array(cap);
    ni.set(this.idx.subarray(0, this.icount));
    this.idx = ni;
  }
  vertex(x, y, z, nx, ny, nz) {
    this._growV(1);
    const v = this.vcount++;
    const fo = v * 8, bo = v * STRIDE;
    this.f[fo] = x; this.f[fo + 1] = y; this.f[fo + 2] = z;
    const l = Math.hypot(nx, ny, nz) || 1;
    this.i8[bo + 12] = Math.round((nx / l) * 127); this.i8[bo + 13] = Math.round((ny / l) * 127); this.i8[bo + 14] = Math.round((nz / l) * 127); this.i8[bo + 15] = 0;
    this.b[bo + 16] = this.col[0]; this.b[bo + 17] = this.col[1]; this.b[bo + 18] = this.col[2]; this.b[bo + 19] = this.col[3];
    this.b[bo + 20] = this.mat[0]; this.b[bo + 21] = this.mat[1]; this.b[bo + 22] = this.mat[2]; this.b[bo + 23] = this.mat[3];
    this.f[fo + 6] = this.aux[0]; this.f[fo + 7] = this.aux[1];
    if (x < this.min[0]) this.min[0] = x; if (y < this.min[1]) this.min[1] = y; if (z < this.min[2]) this.min[2] = z;
    if (x > this.max[0]) this.max[0] = x; if (y > this.max[1]) this.max[1] = y; if (z > this.max[2]) this.max[2] = z;
    return v;
  }
  tri(a, b, c) { this._growI(3); this.idx[this.icount++] = a; this.idx[this.icount++] = b; this.idx[this.icount++] = c; }
  // flat quad a,b,c,d (counter-clockwise seen from the front)
  quad(a, b, c, d) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = d[0] - a[0], vy = d[1] - a[1], vz = d[2] - a[2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const i0 = this.vertex(a[0], a[1], a[2], nx, ny, nz);
    const i1 = this.vertex(b[0], b[1], b[2], nx, ny, nz);
    const i2 = this.vertex(c[0], c[1], c[2], nx, ny, nz);
    const i3 = this.vertex(d[0], d[1], d[2], nx, ny, nz);
    this.tri(i0, i2, i1); this.tri(i0, i3, i2);
  }
  // oriented box: centre, half extents along local axes (u = right, v = up, w = forward)
  box(c, u, v, w, hu, hv, hw, skipBottom = false) {
    const P = (su, sv, sw) => [c[0] + u[0] * hu * su + v[0] * hv * sv + w[0] * hw * sw, c[1] + u[1] * hu * su + v[1] * hv * sv + w[1] * hw * sw, c[2] + u[2] * hu * su + v[2] * hv * sv + w[2] * hw * sw];
    const p = [P(-1, -1, -1), P(1, -1, -1), P(1, 1, -1), P(-1, 1, -1), P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)];
    this.quad(p[0], p[3], p[2], p[1]); // -w
    this.quad(p[5], p[6], p[7], p[4]); // +w
    this.quad(p[4], p[7], p[3], p[0]); // -u
    this.quad(p[1], p[2], p[6], p[5]); // +u
    this.quad(p[3], p[7], p[6], p[2]); // +v
    if (!skipBottom) this.quad(p[4], p[0], p[1], p[5]); // -v
  }
  // axis-aligned-ish box between two points with a given cross-section (beam/strut)
  beam(a, b, width, height, upHint = [0, 1, 0]) {
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L = Math.hypot(dx, dy, dz);
    if (L < 1e-4) return;
    const w = [dx / L, dy / L, dz / L];
    let u = cross(upHint, w);
    if (Math.hypot(...u) < 1e-3) u = cross([1, 0, 0], w);
    u = norm(u);
    const v = cross(w, u);
    this.box([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], u, v, w, width / 2, height / 2, L / 2);
  }
  // round tube between two points
  tube(a, b, r, sides = 6, caps = false) {
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L = Math.hypot(dx, dy, dz);
    if (L < 1e-5) return;
    const w = [dx / L, dy / L, dz / L];
    let u = cross(Math.abs(w[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0], w);
    u = norm(u);
    const v = cross(w, u);
    const base = this.vcount;
    for (let k = 0; k <= sides; k++) {
      const t = (k / sides) * Math.PI * 2;
      const c = Math.cos(t), s = Math.sin(t);
      const n = [u[0] * c + v[0] * s, u[1] * c + v[1] * s, u[2] * c + v[2] * s];
      this.vertex(a[0] + n[0] * r, a[1] + n[1] * r, a[2] + n[2] * r, ...n);
      this.vertex(b[0] + n[0] * r, b[1] + n[1] * r, b[2] + n[2] * r, ...n);
    }
    for (let k = 0; k < sides; k++) {
      const i = base + k * 2;
      this.tri(i, i + 1, i + 3); this.tri(i, i + 3, i + 2);
    }
    if (caps) {
      const ca = this.vertex(a[0], a[1], a[2], -w[0], -w[1], -w[2]);
      const cb = this.vertex(b[0], b[1], b[2], w[0], w[1], w[2]);
      for (let k = 0; k < sides; k++) {
        const t0 = (k / sides) * Math.PI * 2, t1 = ((k + 1) / sides) * Math.PI * 2;
        const pa = (t) => [a[0] + (u[0] * Math.cos(t) + v[0] * Math.sin(t)) * r, a[1] + (u[1] * Math.cos(t) + v[1] * Math.sin(t)) * r, a[2] + (u[2] * Math.cos(t) + v[2] * Math.sin(t)) * r];
        const pb = (t) => [b[0] + (u[0] * Math.cos(t) + v[0] * Math.sin(t)) * r, b[1] + (u[1] * Math.cos(t) + v[1] * Math.sin(t)) * r, b[2] + (u[2] * Math.cos(t) + v[2] * Math.sin(t)) * r];
        const x0 = pa(t0), x1 = pa(t1), y0 = pb(t0), y1 = pb(t1);
        this.tri(ca, this.vertex(...x1, -w[0], -w[1], -w[2]), this.vertex(...x0, -w[0], -w[1], -w[2]));
        this.tri(cb, this.vertex(...y0, ...w), this.vertex(...y1, ...w));
      }
    }
  }
  // surface of revolution around the y axis at (cx, cz): profile = [[r, y], ...] bottom to top
  lathe(cx, by, cz, profile, sides = 24, ellipse = [1, 1]) {
    const base = this.vcount;
    for (let k = 0; k <= sides; k++) {
      const t = (k / sides) * Math.PI * 2;
      const c = Math.cos(t), s = Math.sin(t);
      for (let j = 0; j < profile.length; j++) {
        const [r, y] = profile[j];
        const pj = profile[Math.max(0, j - 1)], nj = profile[Math.min(profile.length - 1, j + 1)];
        const dr = nj[0] - pj[0], dy = nj[1] - pj[1];
        // outward normal of the profile curve
        const nr = dy, ny = -dr;
        this.vertex(cx + c * r * ellipse[0], by + y, cz + s * r * ellipse[1], c * nr, ny, s * nr);
      }
    }
    const P = profile.length;
    for (let k = 0; k < sides; k++) {
      for (let j = 0; j < P - 1; j++) {
        const a = base + k * P + j, b = base + (k + 1) * P + j;
        this.tri(a, a + 1, b + 1); this.tri(a, b + 1, b);
      }
    }
  }
  result() {
    return {
      vertices: new Uint8Array(this.buf, 0, this.vcount * STRIDE).slice(),
      indices: this.idx.slice(0, this.icount),
      vcount: this.vcount, icount: this.icount,
      min: this.min.slice(), max: this.max.slice(),
    };
  }
}

export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
