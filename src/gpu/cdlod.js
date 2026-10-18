// CDLOD quadtree node selection (CPU, allocation free), from web-ocean (MIT) generalised
// for both the water surface and the land, and for mirrored (reflection) cameras.
export class CDLOD {
  constructor({ leaf, grid, lods, maxDist, yMin, yMax, xyMargin, maxNodes }) {
    Object.assign(this, { leaf, grid, lods, maxDist, yMin, yMax, xyMargin, maxNodes });
    this.r0 = 8 * leaf;
    this.ranges = new Float64Array(lods);
    for (let l = 0; l < lods; l++) this.ranges[l] = this.r0 * Math.pow(2, l);
    this.data = new Float32Array(maxNodes * 4);
    this.planes = new Float64Array(16);
    this.count = 0;
    this.cx = 0; this.cy = 0; this.cz = 0;
    this.bounds = null; // optional [x0, z0, x1, z1] limit
  }
  sphereHit(x0, z0, size, r) {
    const dx = Math.max(x0 - this.cx, 0, this.cx - (x0 + size));
    const dz = Math.max(z0 - this.cz, 0, this.cz - (z0 + size));
    const dy = Math.max(this.yMin - this.cy, 0, this.cy - this.yMax);
    return dx * dx + dy * dy + dz * dz <= r * r;
  }
  inFrustum(x0, z0, size) {
    const m = this.xyMargin;
    const minx = x0 - m - this.cx, maxx = x0 + size + m - this.cx;
    const minz = z0 - m - this.cz, maxz = z0 + size + m - this.cz;
    const miny = this.yMin - this.cy, maxy = this.yMax - this.cy;
    const P = this.planes;
    for (let p = 0; p < 16; p += 4) {
      const a = P[p], b = P[p + 1], c = P[p + 2], d = P[p + 3];
      const px = a > 0 ? maxx : minx, py = b > 0 ? maxy : miny, pz = c > 0 ? maxz : minz;
      if (a * px + b * py + c * pz + d < 0) return false;
    }
    if (this.bounds) {
      const B = this.bounds;
      if (x0 > B[2] || z0 > B[3] || x0 + size < B[0] || z0 + size < B[1]) return false;
    }
    return true;
  }
  add(x0, z0, size, lod) {
    if (this.count >= this.maxNodes) return;
    const o = this.count * 4;
    this.data[o] = x0; this.data[o + 1] = z0; this.data[o + 2] = size; this.data[o + 3] = lod;
    this.count++;
  }
  node(x0, z0, size, lod) {
    if (!this.sphereHit(x0, z0, size, this.ranges[lod])) return false;
    if (!this.sphereHit(x0, z0, size, this.maxDist) || !this.inFrustum(x0, z0, size)) return true;
    if (lod === 0 || !this.sphereHit(x0, z0, size, this.ranges[lod - 1])) { this.add(x0, z0, size, lod); return true; }
    const h = size * 0.5;
    for (let i = 0; i < 4; i++) {
      const nx = x0 + (i & 1) * h, nz = z0 + (i >> 1) * h;
      if (!this.node(nx, nz, h, lod - 1)) {
        if (this.sphereHit(nx, nz, h, this.maxDist) && this.inFrustum(nx, nz, h)) this.add(nx, nz, h, lod - 1);
      }
    }
    return true;
  }
  // m: camera-relative view-projection (column-major Float32Array/Float64Array)
  select(cam, m) {
    this.count = 0;
    this.cx = cam[0]; this.cy = cam[1]; this.cz = cam[2];
    const P = this.planes;
    for (let i = 0; i < 4; i++) {
      const r = i >> 1, s = (i & 1) ? -1 : 1;
      P[i * 4] = m[3] + s * m[r];
      P[i * 4 + 1] = m[7] + s * m[4 + r];
      P[i * 4 + 2] = m[11] + s * m[8 + r];
      P[i * 4 + 3] = m[15] + s * m[12 + r];
    }
    const top = this.lods - 1;
    const ts = this.leaf * Math.pow(2, top);
    const x0 = Math.floor((this.cx - this.maxDist) / ts), x1 = Math.floor((this.cx + this.maxDist) / ts);
    const z0 = Math.floor((this.cz - this.maxDist) / ts), z1 = Math.floor((this.cz + this.maxDist) / ts);
    for (let zi = z0; zi <= z1; zi++) for (let xi = x0; xi <= x1; xi++) this.node(xi * ts, zi * ts, ts, top);
    return this.count;
  }
}

export function gridIndices(grid) {
  const idx = new Uint16Array(grid * grid * 6);
  let k = 0;
  for (let j = 0; j < grid; j++) {
    for (let i = 0; i < grid; i++) {
      const a = i + j * (grid + 1), b = a + 1, c = a + grid + 1, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = d;
      idx[k++] = a; idx[k++] = d; idx[k++] = b;
    }
  }
  return idx;
}
