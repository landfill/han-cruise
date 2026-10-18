// Boat wake in route coordinates.
// The boat follows a known path, so every water point near it is mapped once (on the GPU at load)
// to (s, d): arc length of the nearest path point and signed lateral distance. A point at
// x = s_boat - s behind the boat sees the Kelvin wake of a source moving at speed U, evaluated with
// the stationary-phase solution (transverse + divergent systems inside the 19.47 degree wedge,
// caustic at its edge). Bow and stern sources interfere; amplitude decays with the time since the
// boat passed (age). Curved segments of the route curve the wake with them.
// Turbulent stern wash (foam) and the calm, ripple-free stripe behind the boat use the same coordinates.

export const PATH_FIELD = /* wgsl */ `
struct PF { origin: vec4f, info: vec4f };   // origin.xy = x0, z0; origin.z = cell; info.x = sample count, info.y = step
@group(0) @binding(0) var<uniform> U: PF;
@group(0) @binding(1) var<storage, read> pts: array<vec2f>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(dst);
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let p = U.origin.xy + (vec2f(id.xy) + 0.5) * U.origin.z;
  let n = u32(U.info.x);
  var best = 1e9;
  var bs = 0.0;
  var bd = 0.0;
  var bt = vec2f(0.0, -1.0);
  for (var i = 0u; i + 1u < n; i++) {
    let a = pts[i];
    let b = pts[i + 1u];
    let ab = b - a;
    let l2 = max(dot(ab, ab), 1e-6);
    let t = clamp(dot(p - a, ab) / l2, 0.0, 1.0);
    let q = a + ab * t;
    let dd = distance(p, q);
    if (dd < best) {
      best = dd;
      bs = (f32(i) + t) * U.info.y;
      // signed: positive to the starboard (right) of the direction of travel
      let fwd = ab / sqrt(l2);
      let rt = vec2f(-fwd.y, fwd.x);
      bd = dot(p - q, rt);
      bt = fwd;
    }
  }
  textureStore(dst, id.xy, vec4f(bs, bd, bt));
}
`;

// Wake functions, included by the water shader. Requires: F (Frame), pathTex, passTime.
export const KELVIN_WGSL = /* wgsl */ `
const GRAV_W: f32 = 9.81;

// (s, d, tangent) at a world point, manual bilinear (rgba32float is not filterable everywhere)
fn pathCoord(xz: vec2f) -> vec4f {
  let uv = (xz - F.pathP.xy) / F.pathP.z;   // in texels (pathP.z = cell size)
  let dims = vec2i(textureDimensions(pathTex));
  let f = uv - 0.5;
  let i = vec2i(floor(f));
  if (any(i < vec2i(0)) || any(i >= dims - 1)) { return vec4f(0.0, 1e4, 0.0, -1.0); }
  let t = f - floor(f);
  let a = textureLoad(pathTex, i, 0);
  let b = textureLoad(pathTex, i + vec2i(1, 0), 0);
  let c = textureLoad(pathTex, i + vec2i(0, 1), 0);
  let d = textureLoad(pathTex, i + vec2i(1, 1), 0);
  // across the medial axis of a turn neighbouring texels can map to distant parts of the route:
  // fall back to the nearest texel there
  let spread = max(max(abs(a.x - b.x), abs(a.x - c.x)), abs(a.x - d.x));
  var v = mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
  if (spread > F.pathP.z * 3.0) { v = textureLoad(pathTex, vec2i(round(f)), 0); }
  return v;
}
fn passAge(s: f32) -> f32 {
  // time since the boat's centre passed arc length s (passTime is written by the CPU each frame)
  let n = arrayLength(&passTime);
  let i = u32(clamp(s / F.wakeP.z, 0.0, f32(n - 1u)));
  let pt = passTime[i];
  if (pt < -1e5) { return 1e6; }
  return max(F.camPos.w - pt, 0.0);
}

struct Wave { h: f32, gx: f32, gy: f32 };

// Kelvin wake of a point source, stationary phase. x: distance behind the source (> 0), y: lateral.
// With t = |tan(theta)| of a stationary wave direction, |y|/x = t / (1 + 2 t^2):
//   phase  = kU x (1 - T t) / cos(theta)
//   phase'' = kU x (1 - 2t^2)(1 + t^2) / ((1 + 2t^2) cos(theta))   (zero at the 19.47 deg caustic)
fn kelvin(x: f32, y: f32, U: f32) -> Wave {
  var w: Wave;
  w.h = 0.0; w.gx = 0.0; w.gy = 0.0;
  if (x < 0.5 || U < 0.3) { return w; }
  let kU = GRAV_W / (U * U);
  let T = abs(y) / x;
  let sy = select(1.0, -1.0, y < 0.0);
  let Tc = 0.35355;
  let edge = 1.0 - smoothstep(Tc * 0.94, Tc * 1.15, T);
  if (edge <= 0.0) { return w; }
  let Tq = min(T, Tc * 0.998);
  let disc = sqrt(max(1.0 - 8.0 * Tq * Tq, 0.0));
  for (var k = 0; k < 2; k++) {
    var t: f32;
    if (k == 0) { t = select((1.0 - disc) / (4.0 * Tq), Tq, Tq < 1e-3); }
    else {
      if (Tq < 1e-3) { continue; }
      t = (1.0 + disc) / (4.0 * Tq);
    }
    let c2 = 1.0 / (1.0 + t * t);
    let c = sqrt(c2);
    let ph = kU * x * (1.0 - Tq * t) / c;
    let curv = max(abs(kU * x * (1.0 - 2.0 * t * t) * (1.0 + t * t) / ((1.0 + 2.0 * t * t) * c)), 0.8);
    // hull filtering: the short divergent waves of a long hull are weaker
    let filt = select(1.0, 0.65 * exp(-0.18 * t * t), k == 1);
    let amp = sqrt(6.2832 / curv) * filt;
    let pshift = select(0.7854, -0.7854, k == 1);
    let sv = sin(ph + pshift);
    let cv = cos(ph + pshift);
    let kk = kU / c2;
    w.h += amp * sv;
    w.gx += amp * cv * kk * c;
    w.gy += amp * cv * kk * (-t * c) * sy;
  }
  w.h *= edge; w.gx *= edge; w.gy *= edge;
  return w;
}

// Full wake at a world point: x = elevation, yz = world-space slope (d h / d x, d h / d z),
// w = -, plus foam / calm returned through the out parameter.
struct WakeOut { h: f32, slope: vec2f, foam: f32, calm: f32 };
fn boatWake(xz: vec2f, withSlope: bool) -> WakeOut {
  var o: WakeOut;
  o.h = 0.0; o.slope = vec2f(0.0); o.foam = 0.0; o.calm = 0.0;
  let pc = pathCoord(xz);
  let d = pc.y;
  if (abs(d) > 260.0) { return o; }
  let U = F.wakeP.y;
  let L = F.boat2.y;
  let B = F.boat2.z;
  let xb = F.wakeP.x + L * 0.42 - pc.x;   // behind the bow (forward entry point)
  let xs = F.wakeP.x - L * 0.45 - pc.x;   // behind the stern
  if (xb < -5.0) { return o; }
  let age = passAge(pc.x);
  let fade = exp(-age / 50.0);
  let A = F.foamP.y * min(1.0, U * U / 21.0) * fade;
  if (A > 1e-4) {
    let wb = kelvin(xb, d, U);
    let ws = kelvin(xs, d, U);
    o.h = (wb.h - 0.55 * ws.h) * A;
    if (withSlope) {
      // path frame: x grows backwards along the tangent, y to the right of it
      let tg = pc.zw;
      let rt = vec2f(-tg.y, tg.x);
      let gx = (wb.gx - 0.55 * ws.gx) * A;
      let gy = (wb.gy - 0.55 * ws.gy) * A;
      o.slope = -tg * gx + rt * gy;
    }
  }
  // turbulent wash aft of the transom and the calm (ripple-damped) stripe it leaves
  if (xs > -2.0) {
    let wdt = B * 0.42 + max(xs, 0.0) * 0.045;
    let lat = abs(d);
    let core = 1.0 - smoothstep(wdt * 0.55, wdt, lat);
    let sk = smoothstep(0.4, 2.5, U);
    o.foam = core * exp(-age / 16.0) * smoothstep(-2.0, 1.0, xs) * sk;
    o.calm = (1.0 - smoothstep(wdt * 0.8, wdt * 1.4, lat)) * exp(-age / 75.0) * smoothstep(-2.0, 2.0, xs) * sk;
  }
  // bow wave whitening along the forward shoulders at speed
  let bowU = xb;
  if (bowU > -1.0 && bowU < L * 0.5) {
    let sh = exp(-pow(abs(d) - B * 0.5 - 0.6, 2.0) / 0.8) * exp(-bowU / 9.0);
    o.foam = max(o.foam, sh * smoothstep(3.2, 5.5, U) * 0.6);
  }
  return o;
}
`;
