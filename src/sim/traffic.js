// Night traffic on the bridges and riverside expressways: cars move along lane polylines;
// each car emits a white headlight pair (towards its heading) and red tail lights (behind).
// The lights are directional sprites, so oncoming and departing traffic read correctly.
export class Traffic {
  constructor(lanes, seed = 7) {
    let rs = seed;
    const rnd = () => { rs = (Math.imul(rs, 1664525) + 1013904223) >>> 0; return rs / 4294967296; };
    this.lanes = [];
    for (const l of lanes) {
      const pts = l.pts;
      if (pts.length < 2) continue;
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][2] - pts[i - 1][2]));
      const L = cum[cum.length - 1];
      if (L < 60) continue;
      const lane = { pts, cum, L, cars: [] };
      const vmax = l.speed || 16;
      // spacing 30-90 m (moderate evening traffic)
      for (let s = rnd() * 40; s < L; s += 30 + rnd() * 60) {
        const bus = rnd() < 0.06;
        lane.cars.push({ s, v: vmax * (bus ? 0.75 : 0.8 + 0.3 * rnd()), bus, colour: Math.floor(rnd() * 1000) });
      }
      this.lanes.push(lane);
    }
    this.count = this.lanes.reduce((n, l) => n + l.cars.length, 0);
    this.data = new Float32Array(Math.max(1, this.count * 2 + 64) * 8);
    // instance data for car bodies near the camera: x, y, z, yaw, r, g, b, 1
    this.cars = new Float32Array(Math.max(1, this.count) * 8);
    this.buses = new Float32Array(Math.max(1, this.count) * 8);
    this.carCount = 0; this.busCount = 0;
  }
  bodies(cam, maxDist, carCols, busCols) {
    let nc = 0, nb = 0;
    const md2 = maxDist * maxDist;
    for (const lane of this.lanes) for (const c of lane.cars) {
      if (c.x === undefined) continue;
      const dx = c.x - cam[0], dz = c.z - cam[2];
      if (dx * dx + dz * dz > md2) continue;
      const arr = c.bus ? this.buses : this.cars;
      const o = (c.bus ? nb++ : nc++) * 8;
      const col = c.bus ? busCols[c.colour % busCols.length] : carCols[c.colour % carCols.length];
      arr[o] = c.x; arr[o + 1] = c.y; arr[o + 2] = c.z; arr[o + 3] = c.yaw;
      arr[o + 4] = col[0]; arr[o + 5] = col[1]; arr[o + 6] = col[2]; arr[o + 7] = 1;
    }
    this.carCount = nc; this.busCount = nb;
  }
  // writes sprite instances (x,y,z,size, r,g,b,yaw) and returns the count
  update(dt, extra, lightsOn = true) {
    const d = this.data;
    let n = 0;
    for (const lane of this.lanes) {
      const { pts, cum, L } = lane;
      for (const c of lane.cars) {
        c.s = (c.s + c.v * dt) % L;
        let i = 1;
        // linear search from a cached index (cars move forward)
        i = c.i && cum[c.i - 1] <= c.s ? c.i : 1;
        while (i < cum.length - 1 && cum[i] < c.s) i++;
        c.i = i;
        const t = (c.s - cum[i - 1]) / Math.max(cum[i] - cum[i - 1], 1e-6);
        const a = pts[i - 1], b = pts[i];
        const x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t, z = a[2] + (b[2] - a[2]) * t;
        const dx = b[0] - a[0], dz = b[2] - a[2];
        const len = Math.hypot(dx, dz) || 1;
        const yaw = Math.atan2(dx, -dz);
        const fx = dx / len, fz = dz / len;
        c.x = x; c.y = y; c.z = z; c.yaw = yaw;
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
    if (extra) for (let i = 0; i < extra.length; i++) d[n * 8 + i] = extra[i];
    if (extra) n += extra.length / 8;
    return n;
  }
}
