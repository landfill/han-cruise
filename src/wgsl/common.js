// Shared constants, the per-frame uniform layout and WGSL helpers.
// Portions adapted from web-ocean (c) 2026 Maksymilian Dendura, MIT licence
// (see THIRD_PARTY_NOTICES.md): ATMO_CORE_WGSL, NOISE_WGSL, PERIODIC_WGSL.

export const CASCADES = 4;
export const SKY_W = 256;      // sky-view LUT size
export const SKY_H = 144;
export const ENV_SIZE = 128;   // environment cube face size
export const ENV_MIPS = 8;
export const WAKE_RES = 512;   // boat wake simulation grid
export const WAKE_SIZE = 256;  // metres covered by the wake grid

// Per-frame uniform block. Every field is a vec4f or mat4x4f so offsets are trivial;
// FRAME.<name> gives the float offset used by the JS side.
const FIELDS = [
  ['viewProj', 'mat4x4f'],      // camera-relative view-projection (reverse-Z, infinite)
  ['reflViewProj', 'mat4x4f'],  // camera-relative VP of the mirrored (planar reflection) camera
  ['shadowVP0', 'mat4x4f'],     // camera-relative shadow cascade matrices
  ['shadowVP1', 'mat4x4f'],
  ['shadowVP2', 'mat4x4f'],
  ['camPos', 'vec4f'],          // xyz metres, w = time (s)
  ['camRight', 'vec4f'],        // xyz * tan(fovX/2)
  ['camUp', 'vec4f'],           // xyz * tan(fovY/2)
  ['camFwd', 'vec4f'],          // xyz, w = reflection pass flag
  ['sunDir', 'vec4f'],          // xyz towards sun, w = solar illuminance scale
  ['moonDir', 'vec4f'],         // xyz towards moon, w = lunar illuminance (relative)
  ['screen', 'vec4f'],          // w, h, 1/w, 1/h
  ['cascL', 'vec4f'],           // FFT cascade patch sizes (m)
  ['cascInvL', 'vec4f'],
  ['windP', 'vec4f'],           // xy = wave direction (unit), z = wind speed, w = gust strength
  ['current', 'vec4f'],         // xy = surface current (m/s), z = advected time, w = -
  ['lodP', 'vec4f'],            // CDLOD base range, grid quads per node, terrain base range, -
  ['post', 'vec4f'],            // exposure, bloom strength, frame index, debug view
  ['atmo', 'vec4f'],            // haze (mie multiplier), camera altitude km, high cloud cover, high cloud density
  ['waterCol', 'vec4f'],        // body scatter colour rgb, unresolved slope variance
  ['sssCol', 'vec4f'],          // forward scatter colour rgb, strength
  ['foamP', 'vec4f'],           // wake foam intensity, -, -, -
  ['fogExt', 'vec4f'],          // rgb extinction per metre near the ground, w = haze floor
  ['cloudP', 'vec4f'],          // low cloud altitude km, coverage, optical density, wind time
  ['night', 'vec4f'],           // x = darkness 0..1, y = artificial light scale, z = skyglow, w = window lit fraction
  ['boat', 'vec4f'],            // x, z, heading (rad, cw from north), speed m/s
  ['boat2', 'vec4f'],           // yaw rate, hull length, beam, draft
  ['wakeP', 'vec4f'],           // boat arc length s, smoothed speed U, pass-time sample step, -
  ['pathP', 'vec4f'],           // route field origin x, z, cell size, -
  ['shadowP', 'vec4f'],         // 1/shadow map size, cascade 0/1/2 far distances packed in yzw
  ['sdfP', 'vec4f'],            // river SDF texture origin x, z, size x, size z (m)
  ['hgtP', 'vec4f'],            // land height texture origin x, z, size x, size z (m)
  ['farP', 'vec4f'],            // far terrain texture origin x, z, size x, size z (m)
  ['misc', 'vec4f'],            // x = render scale, y = shadow on, z = reflection mip count, w = ambient scale
];
export const FRAME = { floats: 0 };
let off = 0;
let wgsl = 'struct Frame {\n';
for (const [n, t] of FIELDS) {
  FRAME[n] = off;
  off += t === 'mat4x4f' ? 16 : 4;
  wgsl += `  ${n}: ${t},\n`;
}
wgsl += '};\n';
FRAME.floats = off;
export const FRAME_WGSL = wgsl;

export const ATMO_CORE_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const RG: f32 = 6360.0;
const RT: f32 = 6460.0;
const RAY_S: vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const MIE_S: f32 = 3.996e-3;
const MIE_E: f32 = 4.40e-3;
const OZO_A: vec3f = vec3f(0.650e-3, 1.881e-3, 0.085e-3);
const SUN_ANG_R: f32 = 0.004675;
const TRANS_W: f32 = 256.0;
const TRANS_H: f32 = 64.0;
const MS_SIZE: f32 = 32.0;

struct Medium { ext: vec3f, rayS: vec3f, mieS: f32 };

fn medium(h: f32, haze: f32) -> Medium {
  let hh = max(h, 0.0);
  let dR = exp(-hh / 8.0);
  // urban aerosols: a 1.2 km boundary layer
  let dM = exp(-hh / 1.2);
  let dO = max(0.0, 1.0 - abs(hh - 25.0) / 15.0);
  var m: Medium;
  m.rayS = RAY_S * dR;
  m.mieS = MIE_S * dM * haze;
  m.ext = m.rayS + vec3f(MIE_E * dM * haze) + OZO_A * dO;
  return m;
}

fn raySphere(ro: vec3f, rd: vec3f, R: f32) -> vec2f {
  let l = length(ro);
  let b = dot(ro, rd);
  let c = (l - R) * (l + R);
  let d = b * b - c;
  if (d < 0.0) { return vec2f(-1.0, -1.0); }
  let s = sqrt(d);
  return vec2f(-b - s, -b + s);
}

fn rhoOf(r: f32) -> f32 { return sqrt(max(0.0, (r - RG) * (r + RG))); }

fn transRMuToUV(r: f32, mu: f32) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = rhoOf(r);
  let disc = r * r * (mu * mu - 1.0) + RT * RT;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = RT - r;
  let dMax = rho + H;
  let x = (d - dMin) / (dMax - dMin);
  let y = rho / H;
  return vec2f(x, y);
}

fn transUVToRMu(uv: vec2f) -> vec2f {
  let H = sqrt(RT * RT - RG * RG);
  let rho = H * uv.y;
  let r = sqrt(rho * rho + RG * RG);
  let dMin = RT - r;
  let dMax = rho + H;
  let d = dMin + uv.x * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

fn fromUnitToSubUvs(u: f32, res: f32) -> f32 { return (u + 0.5 / res) * (res / (res + 1.0)); }
fn fromSubUvsToUnit(u: f32, res: f32) -> f32 { return (u - 0.5 / res) * (res / (res - 1.0)); }

fn phaseRayleigh(c: f32) -> f32 { return 3.0 / (16.0 * PI) * (1.0 + c * c); }
fn phaseMie(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return 3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + c * c)) / ((2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}
fn phaseHG(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

fn sunBelowHorizon(r: f32, mu: f32) -> bool {
  return mu < -rhoOf(r) / r;
}
`;

export const NOISE_WGSL = /* wgsl */ `
fn pcgHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hashGrad(i: vec2i) -> vec2f {
  let h = pcgHash(u32(i.x) * 1664525u + pcgHash(u32(i.y) + 1013904223u));
  let a = f32(h) * (6.28318530718 / 4294967296.0);
  return vec2f(cos(a), sin(a));
}
fn gnoise(p: vec2f) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = dot(hashGrad(i), f);
  let b = dot(hashGrad(i + vec2i(1, 0)), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i + vec2i(0, 1)), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i + vec2i(1, 1)), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn hash21(p: vec2f) -> f32 {
  let h = pcgHash((u32(i32(floor(p.x))) * 73856093u) ^ pcgHash(u32(i32(floor(p.y))) * 19349663u));
  return f32(h >> 8u) / 16777216.0;
}
fn hashU(a: u32, b: u32) -> f32 { return f32(pcgHash((a * 2654435761u) ^ pcgHash(b)) >> 8u) / 16777216.0; }
`;

export const PERIODIC_WGSL = /* wgsl */ `
fn pnoise(p: vec2f, period: i32) -> f32 {
  let i = vec2i(floor(p));
  let f = p - floor(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let i00 = (i % period + period) % period;
  let i10 = ((i + vec2i(1, 0)) % period + period) % period;
  let i01 = ((i + vec2i(0, 1)) % period + period) % period;
  let i11 = ((i + vec2i(1, 1)) % period + period) % period;
  let a = dot(hashGrad(i00), f);
  let b = dot(hashGrad(i10), f - vec2f(1.0, 0.0));
  let c = dot(hashGrad(i01), f - vec2f(0.0, 1.0));
  let d = dot(hashGrad(i11), f - vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn pfbm(p: vec2f, period: i32, oct: i32) -> f32 {
  var s = 0.0; var a = 0.5; var q = p; var per = period;
  for (var o = 0; o < oct; o++) {
    s += a * pnoise(q, per);
    q *= 2.0; per *= 2; a *= 0.5;
  }
  return s;
}
`;
