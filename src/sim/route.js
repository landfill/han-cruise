// Cruise route: a smooth arc-length parameterised path through waypoints, a speed
// profile (gentle departure, cruise, slow turn, approach), and the boat's motion along it.
// World frame: x = east, y = up, z = south. Heading: radians clockwise from north.

export class Path {
  constructor(points, step = 2) {
    // centripetal Catmull-Rom through the waypoints, resampled every `step` metres
    const P = points;
    const dense = [];
    for (let i = 0; i < P.length - 1; i++) {
      const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)];
      const n = Math.max(4, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 1));
      for (let k = 0; k < n; k++) dense.push(catmull(p0, p1, p2, p3, k / n));
    }
    dense.push(P[P.length - 1]);
    const cum = [0];
    for (let i = 1; i < dense.length; i++) cum.push(cum[i - 1] + Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]));
    this.length = cum[cum.length - 1];
    const n = Math.ceil(this.length / step) + 1;
    this.step = this.length / (n - 1);
    this.x = new Float64Array(n);
    this.z = new Float64Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const s = i * this.step;
      while (j < cum.length - 2 && cum[j + 1] < s) j++;
      const t = (s - cum[j]) / Math.max(cum[j + 1] - cum[j], 1e-9);
      this.x[i] = dense[j][0] + (dense[j + 1][0] - dense[j][0]) * Math.min(1, t);
      this.z[i] = dense[j][1] + (dense[j + 1][1] - dense[j][1]) * Math.min(1, t);
    }
    // heading and curvature per sample (smoothed)
    this.heading = new Float64Array(n);
    this.curv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 2), b = Math.min(n - 1, i + 2);
      this.heading[i] = Math.atan2(this.x[b] - this.x[a], -(this.z[b] - this.z[a]));
    }
    for (let i = 1; i < n; i++) {
      let d = this.heading[i] - this.heading[i - 1];
      d -= Math.round(d / (2 * Math.PI)) * 2 * Math.PI;
      this.heading[i] = this.heading[i - 1] + d; // unwrapped
    }
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - 6), b = Math.min(n - 1, i + 6);
      this.curv[i] = (this.heading[b] - this.heading[a]) / Math.max((b - a) * this.step, 1e-6);
    }
  }
  sample(s, out) {
    const f = Math.max(0, Math.min(this.length, s)) / this.step;
    const i = Math.min(Math.floor(f), this.x.length - 2);
    const t = f - i;
    out.x = this.x[i] + (this.x[i + 1] - this.x[i]) * t;
    out.z = this.z[i] + (this.z[i + 1] - this.z[i]) * t;
    out.heading = this.heading[i] + (this.heading[i + 1] - this.heading[i]) * t;
    out.curv = this.curv[i] + (this.curv[i + 1] - this.curv[i]) * t;
    return out;
  }
}

function catmull(p0, p1, p2, p3, t) {
  const alpha = 0.5;
  const tj = (a, b) => Math.pow(Math.hypot(b[0] - a[0], b[1] - a[1]) || 1e-3, alpha);
  const t0 = 0, t1 = t0 + tj(p0, p1), t2 = t1 + tj(p1, p2), t3 = t2 + tj(p2, p3);
  const u = t1 + (t2 - t1) * t;
  const lerp = (a, b, ta, tb) => {
    const w = (u - ta) / (tb - ta || 1e-9);
    return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w];
  };
  const A1 = lerp(p0, p1, t0, t1), A2 = lerp(p1, p2, t1, t2), A3 = lerp(p2, p3, t2, t3);
  const B1 = lerp(A1, A2, t0, t2), B2 = lerp(A2, A3, t1, t3);
  return lerp(B1, B2, t1, t2);
}

// Boat kinematics along the path. Speed follows a target profile with limited
// acceleration; heading lags the path tangent slightly (a real hull yaws, then tracks).
// the approach profile brings the boat to rest 3 m before the end of the path; the voyage counts
// as finished once it has stopped within this distance
const BERTH = 3.5;

export class Boat {
  constructor(path, opts) {
    this.path = path;
    this.cruise = opts.cruise;          // m/s
    this.turnSpeed = opts.turnSpeed;    // m/s through tight turns
    this.accel = opts.accel;            // m/s^2
    this.s = 0;
    this.v = 0;
    this.heading = path.heading[0];
    this.yawRate = 0;
    this.roll = 0; this.pitch = 0; this.heave = 0;
    this.x = path.x[0]; this.z = path.z[0];
    this.finished = false;
    this._tmp = { x: 0, z: 0, heading: 0, curv: 0 };
    this._ahead = { x: 0, z: 0, heading: 0, curv: 0 };
  }
  targetSpeed(s) {
    const L = this.path.length;
    // look ahead for curvature so the boat slows before turns
    let k = 0;
    for (let d = 0; d <= 120; d += 10) k = Math.max(k, Math.abs(this.path.sample(s + d, this._ahead).curv));
    const turn = Math.min(1, k / 0.012);
    let v = this.cruise + (this.turnSpeed - this.cruise) * turn;
    // approach: decelerate to a stop at the end (v^2 = 2 a d)
    const dEnd = L - s;
    v = Math.min(v, Math.sqrt(Math.max(0, 2 * this.accel * 0.6 * (dEnd - 3))));
    // berthing: dead slow over the last ~150 m (about 1 m/s at 20 m out)
    v = Math.min(v, 0.45 + Math.max(0, dEnd - 3) * 0.028);
    return v;
  }
  // Sets the boat at arc length s in steady state (used for reproducible captures).
  place(s) {
    this.s = Math.max(0, Math.min(this.path.length, s));
    const p = this.path.sample(this.s, this._tmp);
    this.v = Math.min(this.targetSpeed(this.s), this.s < 1 ? 0 : 1e9);
    this.heading = p.heading; this.x = p.x; this.z = p.z;
    this.yawRate = p.curv * this.v;
    this.finished = this.s >= this.path.length - BERTH;
  }
  update(dt, running) {
    const vt = running && !this.finished ? this.targetSpeed(this.s) : 0;
    const dv = vt - this.v;
    const a = dv > 0 ? this.accel : this.accel * 1.4;
    this.v += Math.sign(dv) * Math.min(Math.abs(dv), a * dt);
    this.s += this.v * dt;
    if (this.s >= this.path.length - BERTH && this.v < 0.05) this.finished = true;
    const p = this.path.sample(this.s, this._tmp);
    this.x = p.x; this.z = p.z;
    // heading: critically damped follow of the path tangent
    const target = p.heading;
    const err = target - this.heading;
    const w = 1.6;
    const acc = w * w * err - 2 * w * this.yawRate;
    this.yawRate += acc * dt;
    this.heading += this.yawRate * dt;
  }
  // Small, damped hull motion: river chop, heel in turns, bow-up trim with speed.
  motion(time, reduced) {
    const k = reduced ? 0.0 : 1.0;
    const chop = 0.0045 * Math.sin(time * 1.9) + 0.003 * Math.sin(time * 3.1 + 1.3) + 0.0015 * Math.sin(time * 5.3 + 0.4);
    const pc = 0.0035 * Math.sin(time * 1.37 + 0.7) + 0.0022 * Math.sin(time * 2.6 + 2.1);
    // outward heel of a displacement hull in a turn: ~ v * yawRate
    const heel = -Math.max(-0.035, Math.min(0.035, this.v * this.yawRate * 0.18));
    const trim = -0.006 * Math.min(1, this.v / 5);
    this.roll = (chop * (0.4 + 0.6 * Math.min(1, this.v / 4)) + heel) * k;
    this.pitch = (pc * 0.7 + trim) * k + (reduced ? trim : 0);
    this.heave = (0.018 * Math.sin(time * 1.6 + 0.3) + 0.01 * Math.sin(time * 2.9)) * k;
  }
}
