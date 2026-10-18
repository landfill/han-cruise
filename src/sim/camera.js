// Deck camera: a standing passenger's eye on the observation deck. The view follows the boat's
// heading; the boat's pitch and roll are only partly passed on (people keep their head level),
// and the eye stays inside the walkable deck area. Viewpoint changes walk/ease between spots,
// or cut instantly when reduced motion is on.
import { transformPoint, D2R, clamp } from '../gpu/math.js';

const ease = (t) => t * t * (3 - 2 * t);
const wrapDeg = (a) => ((a + 540) % 360) - 180;

export class DeckCamera {
  constructor(viewpoints, walk) {
    this.viewpoints = viewpoints;
    this.walk = walk;
    this.fovY = 55 * D2R;
    const v = viewpoints[0];
    this.key = v.key;
    this.local = v.pos.slice();
    this.yaw = v.yaw; this.pitch = v.pitch;
    this.trans = null;
    this.out = { pos: [0, 0, 0], right: [1, 0, 0], up: [0, 1, 0], fwd: [0, 0, -1], fovY: this.fovY };
  }
  setView(key, instant) {
    const v = this.viewpoints.find((p) => p.key === key);
    if (!v) return;
    this.key = key;
    if (instant) {
      this.local = v.pos.slice(); this.yaw = v.yaw; this.pitch = v.pitch; this.trans = null;
      return;
    }
    const dist = Math.hypot(v.pos[0] - this.local[0], v.pos[1] - this.local[1], v.pos[2] - this.local[2]);
    this.trans = {
      from: { pos: this.local.slice(), yaw: this.yaw, pitch: this.pitch },
      to: { pos: v.pos.slice(), yaw: this.yaw + wrapDeg(v.yaw - this.yaw), pitch: v.pitch },
      t: 0, dur: clamp(0.8 + dist / 9, 0.9, 3.2),
    };
  }
  look(dyawDeg, dpitchDeg) {
    if (this.trans) { this.trans.to.yaw += dyawDeg; this.trans.to.pitch = clamp(this.trans.to.pitch + dpitchDeg, -60, 60); }
    this.yaw += dyawDeg;
    this.pitch = clamp(this.pitch + dpitchDeg, -60, 60);
  }
  zoom(f) { this.fovY = clamp(this.fovY * f, 30 * D2R, 75 * D2R); }
  update(dt, boatM, boat, reduced) {
    if (this.trans) {
      const tr = this.trans;
      tr.t += dt / tr.dur;
      const k = ease(Math.min(1, tr.t));
      for (let i = 0; i < 3; i++) this.local[i] = tr.from.pos[i] + (tr.to.pos[i] - tr.from.pos[i]) * k;
      this.yaw = tr.from.yaw + (tr.to.yaw - tr.from.yaw) * k;
      this.pitch = tr.from.pitch + (tr.to.pitch - tr.from.pitch) * k;
      if (tr.t >= 1) this.trans = null;
    }
    // keep the eye on the deck (inside the railing)
    const w = this.walk;
    if (this.local[1] > w.y) {
      this.local[2] = clamp(this.local[2], w.z0, w.z1);
      const t = (this.local[2] - w.z0) / (w.z1 - w.z0);
      const hw = w.w0 + (w.w1 - w.w0) * t;
      this.local[0] = clamp(this.local[0], -hw, hw);
    }
    const p = transformPoint(boatM, this.local);
    // head stabilisation: keep 25 % of the hull roll/pitch
    const Y = boat.heading + this.yaw * D2R;
    const P = this.pitch * D2R + boat.pitch * 0.25 * Math.cos(this.yaw * D2R) - boat.roll * 0.25 * Math.sin(this.yaw * D2R);
    const roll = boat.roll * 0.25 * Math.cos(this.yaw * D2R) + boat.pitch * 0.25 * Math.sin(this.yaw * D2R);
    const cp = Math.cos(P), sp = Math.sin(P), cy = Math.cos(Y), sy = Math.sin(Y);
    const f = [sy * cp, sp, -cy * cp];
    let r = [cy, 0, sy];
    let u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    // roll about the forward axis
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const r2 = [r[0] * cr + u[0] * sr, r[1] * cr + u[1] * sr, r[2] * cr + u[2] * sr];
    const u2 = [u[0] * cr - r[0] * sr, u[1] * cr - r[1] * sr, u[2] * cr - r[2] * sr];
    void reduced;
    this.out.pos = p; this.out.fwd = f; this.out.right = r2; this.out.up = u2; this.out.fovY = this.fovY;
    return this.out;
  }
}

// Development / appreciation free-flight camera (not the default experience).
export class FreeCamera {
  constructor() { this.pos = [0, 30, 0]; this.yaw = 0; this.pitch = -10; this.fovY = 55 * D2R; this.speed = 20; this.out = { pos: [0, 0, 0], right: [1, 0, 0], up: [0, 1, 0], fwd: [0, 0, -1], fovY: this.fovY }; }
  from(cam) {
    this.pos = cam.pos.slice();
    this.yaw = Math.atan2(cam.fwd[0], -cam.fwd[2]) / D2R;
    this.pitch = Math.asin(clamp(cam.fwd[1], -1, 1)) / D2R;
  }
  look(dy, dp) { this.yaw += dy; this.pitch = clamp(this.pitch + dp, -89, 89); }
  update(dt, keys, minY) {
    const Y = this.yaw * D2R, P = this.pitch * D2R;
    const f = [Math.sin(Y) * Math.cos(P), Math.sin(P), -Math.cos(Y) * Math.cos(P)];
    const r = [Math.cos(Y), 0, Math.sin(Y)];
    const mv = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0), st = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0);
    const up = (keys.KeyE || keys.Space ? 1 : 0) - (keys.KeyQ || keys.KeyC ? 1 : 0);
    const v = this.speed * (keys.ShiftLeft ? 5 : 1) * dt;
    for (let i = 0; i < 3; i++) this.pos[i] += (f[i] * mv + r[i] * st) * v;
    this.pos[1] += up * v;
    // never below the water surface
    this.pos[1] = Math.max(this.pos[1], minY);
    const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    this.out.pos = this.pos.slice(); this.out.fwd = f; this.out.right = r; this.out.up = u; this.out.fovY = this.fovY;
    return this.out;
  }
}
