// Night traffic on the bridges and riverside expressways: cars move along lane polylines;
// each car emits a white headlight pair (towards its heading) and red tail lights (behind).
// The lights are directional sprites, so oncoming and departing traffic read correctly.
// All cars live in flat typed arrays and each lane segment's direction and yaw are computed
// once, so the per-frame update allocates nothing (no boxed numbers for the GC to collect).
export class Traffic {
  constructor(lanes, seed = 7) {
    let rs = seed;
    const rnd = () => { rs = (Math.imul(rs, 1664525) + 1013904223) >>> 0; return rs / 4294967296; };
    this.lanes = [];
    const cars = [];
    for (const l of lanes) {
      const pts = l.pts;
      if (pts.length < 2) continue;
      const n = pts.length;
      const cum = new Float64Array(n);
      for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2]);
      const L = cum[n - 1];
      if (L < 60) continue;
      // points and per-segment unit direction / yaw (segment k runs from point k-1 to point k)
      const px = new Float64Array(n), py = new Float64Array(n), pz = new Float64Array(n);
      const fx = new Float64Array(n), fz = new Float64Array(n), yaw = new Float64Array(n);
      for (let i = 0; i < n; i++) { px[i] = pts[i][0]; py[i] = pts[i][1]; pz[i] = pts[i][2]; }
      for (let i = 1; i < n; i++) {
        const dx = px[i] - px[i - 1], dz = pz[i] - pz[i - 1];
        const len = Math.hypot(dx, dz) || 1;
        fx[i] = dx / len; fz[i] = dz / len; yaw[i] = Math.atan2(dx, -dz);
      }
      const lane = { px, py, pz, fx, fz, yaw, cum, L, first: cars.length, end: 0 };
      const vmax = l.speed || 16;
      // spacing 30-90 m (moderate evening traffic)
      for (let s = rnd() * 40; s < L; s += 30 + rnd() * 60) {
        const bus = rnd() < 0.06;
        cars.push({ s, v: vmax * (bus ? 0.75 : 0.8 + 0.3 * rnd()), bus, colour: Math.floor(rnd() * 1000) });
      }
      lane.end = cars.length;
      this.lanes.push(lane);
    }
    const N = this.count = cars.length;
    this.s = new Float64Array(N); this.v = new Float64Array(N);
    this.seg = new Int32Array(N);   // cached segment index (cars move forward); 0 = not set yet
    this.bus = new Uint8Array(N); this.colour = new Int32Array(N);
    this.x = new Float64Array(N); this.y = new Float64Array(N); this.z = new Float64Array(N); this.yaw = new Float64Array(N);
    cars.forEach((c, k) => { this.s[k] = c.s; this.v[k] = c.v; this.bus[k] = c.bus ? 1 : 0; this.colour[k] = c.colour; });
    this.placed = false;
    this.data = new Float32Array(Math.max(1, this.count * 2 + 64) * 8);
    // instance data for car bodies near the camera: x, y, z, yaw, r, g, b, 1
    this.cars = new Float32Array(Math.max(1, this.count) * 8);
    this.buses = new Float32Array(Math.max(1, this.count) * 8);
    this.carCount = 0; this.busCount = 0;
  }
  bodies(cam, maxDist, carCols, busCols) {
    let nc = 0, nb = 0;
    if (!this.placed) { this.carCount = 0; this.busCount = 0; return; }
    const md2 = maxDist * maxDist;
    const cx = cam[0], cz = cam[2];
    const X = this.x, Y = this.y, Z = this.z, YAW = this.yaw, BUS = this.bus, COL = this.colour;
    for (let k = 0; k < this.count; k++) {
      const dx = X[k] - cx, dz = Z[k] - cz;
      if (dx * dx + dz * dz > md2) continue;
      const isBus = BUS[k] === 1;
      const arr = isBus ? this.buses : this.cars;
      const o = (isBus ? nb++ : nc++) * 8;
      const col = isBus ? busCols[COL[k] % busCols.length] : carCols[COL[k] % carCols.length];
      arr[o] = X[k]; arr[o + 1] = Y[k]; arr[o + 2] = Z[k]; arr[o + 3] = YAW[k];
      arr[o + 4] = col[0]; arr[o + 5] = col[1]; arr[o + 6] = col[2]; arr[o + 7] = 1;
    }
    this.carCount = nc; this.busCount = nb;
  }
  // writes sprite instances (x,y,z,size, r,g,b,yaw) and returns the count
  update(dt, extra, lightsOn = true) {
    const d = this.data;
    const S = this.s, V = this.v, SEG = this.seg, X = this.x, Y = this.y, Z = this.z, YAW = this.yaw;
    let n = 0;
    for (const lane of this.lanes) {
      const { px, py, pz, fx: FX, fz: FZ, yaw: LY, cum, L } = lane;
      const last = cum.length - 1;
      for (let k = lane.first; k < lane.end; k++) {
        const s = (S[k] + V[k] * dt) % L;
        S[k] = s;
        // linear search from the cached segment
        let i = SEG[k] && cum[SEG[k] - 1] <= s ? SEG[k] : 1;
        while (i < last && cum[i] < s) i++;
        SEG[k] = i;
        const t = (s - cum[i - 1]) / Math.max(cum[i] - cum[i - 1], 1e-6);
        const x = px[i - 1] + (px[i] - px[i - 1]) * t, y = py[i - 1] + (py[i] - py[i - 1]) * t, z = pz[i - 1] + (pz[i] - pz[i - 1]) * t;
        const yaw = LY[i], fx = FX[i], fz = FZ[i];
        X[k] = x; Y[k] = y; Z[k] = z; YAW[k] = yaw;
        if (!lightsOn) continue;
        // headlights (front, 2.2 m ahead of the centre)
        let o = n * 8;
        d[o] = x + fx * 2.2; d[o + 1] = y + 0.7; d[o + 2] = z + fz * 2.2; d[o + 3] = 0.22;
        d[o + 4] = 1.0; d[o + 5] = 0.93; d[o + 6] = 0.82; d[o + 7] = yaw; n++;
        o = n * 8;
        d[o] = x - fx * 2.2; d[o + 1] = y + 0.8; d[o + 2] = z - fz * 2.2; d[o + 3] = 0.16;
        d[o + 4] = 0.55; d[o + 5] = 0.02; d[o + 6] = 0.01; d[o + 7] = yaw + Math.PI; n++;
      }
    }
    this.placed = true;
    if (extra) for (let i = 0; i < extra.length; i++) d[n * 8 + i] = extra[i];
    if (extra) n += extra.length / 8;
    return n;
  }
}
