// Physically based atmosphere (Hillaire 2020 style LUTs) + procedural cloud layers.
// Adapted from web-ocean (c) 2026 Maksymilian Dendura, MIT licence (see THIRD_PARTY_NOTICES.md).
// Changes: city skyglow and lit cloud bases at night, moon disc, env cube horizon band shows
// a dim distant city instead of open sea, resolution constants for a lighter GPU budget.
import { FRAME_WGSL, ATMO_CORE_WGSL, NOISE_WGSL, SKY_W, SKY_H, ENV_SIZE, DOME_W, DOME_H, CSHADOW_SIZE } from './common.js';

const ATMO_UNIFORM = /* wgsl */ `
struct AtmoP { haze: vec4f };   // x = mie multiplier
@group(0) @binding(0) var<uniform> A: AtmoP;
`;

export const TRANS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= TRANS_W || f32(id.y) >= TRANS_H) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(TRANS_W, TRANS_H);
  let rm = transUVToRMu(uv);
  let ro = vec3f(0.0, rm.x, 0.0);
  let rd = vec3f(sqrt(max(0.0, 1.0 - rm.y * rm.y)), rm.y, 0.0);
  let tMax = raySphere(ro, rd, RT).y;
  var od = vec3f(0.0);
  let STEPS = 64u;
  let dt = tMax / f32(STEPS);
  for (var i = 0u; i < STEPS; i++) {
    let p = ro + rd * ((f32(i) + 0.5) * dt);
    od += medium(length(p) - RG, A.haze.x).ext * dt;
  }
  textureStore(dst, id.xy, vec4f(exp(-od), 1.0));
}
`;

export const MS_LUT = ATMO_CORE_WGSL + ATMO_UNIFORM + /* wgsl */ `
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var sampC: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= MS_SIZE || f32(id.y) >= MS_SIZE) { return; }
  var uv = (vec2f(id.xy) + 0.5) / MS_SIZE;
  uv = vec2f(fromSubUvsToUnit(uv.x, MS_SIZE), fromSubUvsToUnit(uv.y, MS_SIZE));
  let cosSun = uv.x * 2.0 - 1.0;
  let h = max(uv.y * (RT - RG), 0.01);
  let r = RG + h;
  let ro = vec3f(0.0, r, 0.0);
  let sun = vec3f(sqrt(max(0.0, 1.0 - cosSun * cosSun)), cosSun, 0.0);
  var Lsum = vec3f(0.0);
  var Fsum = vec3f(0.0);
  let SQ = 8u;
  let STEPS = 20u;
  for (var a = 0u; a < SQ; a++) {
    for (var b = 0u; b < SQ; b++) {
      let th = acos(1.0 - 2.0 * (f32(b) + 0.5) / f32(SQ));
      let ph = 2.0 * PI * (f32(a) + 0.5) / f32(SQ);
      let rd = vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
      let tTop = raySphere(ro, rd, RT);
      let tGr = raySphere(ro, rd, RG);
      var tMax = tTop.y;
      let hitG = tGr.x > 0.0;
      if (hitG) { tMax = tGr.x; }
      let dt = tMax / f32(STEPS);
      var T = vec3f(1.0);
      var L = vec3f(0.0);
      var Fms = vec3f(0.0);
      for (var s = 0u; s < STEPS; s++) {
        let p = ro + rd * ((f32(s) + 0.5) * dt);
        let pr = length(p);
        let m = medium(pr - RG, A.haze.x);
        let scat = m.rayS + vec3f(m.mieS);
        let Ts = exp(-m.ext * dt);
        let smu = dot(p, sun) / pr;
        var sunT = sampleTrans(pr, smu);
        if (sunBelowHorizon(pr, smu)) { sunT = vec3f(0.0); }
        let ext = max(m.ext, vec3f(1e-9));
        let S = scat * sunT / (4.0 * PI);
        L += T * (S - S * Ts) / ext;
        Fms += T * (scat - scat * Ts) / ext;
        T *= Ts;
      }
      if (hitG) {
        let pg = ro + rd * tMax;
        let ng = normalize(pg);
        let smu = dot(ng, sun);
        L += T * sampleTrans(RG, smu) * max(smu, 0.0) * 0.06 / PI;
      }
      Lsum += L;
      Fsum += Fms;
    }
  }
  let invN = 1.0 / f32(SQ * SQ);
  let L2 = Lsum * invN;
  let fms = Fsum * invN;
  textureStore(dst, id.xy, vec4f(L2 / (1.0 - fms), 1.0));
}
`;

// Sky-view LUT, recomputed every frame (sun / camera altitude may change).
export const SKYVIEW_LUT = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var msLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= SKY_W || f32(id.y) >= SKY_H) { return; }
  var uv = (vec2f(id.xy) + 0.5) / vec2f(SKY_W, SKY_H);
  uv = vec2f(fromSubUvsToUnit(uv.x, SKY_W), fromSubUvsToUnit(uv.y, SKY_H));
  let haze = F.atmo.x;
  let r = RG + F.atmo.y;
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  var vza: f32;
  if (uv.y < 0.5) {
    var coord = 2.0 * uv.y;
    coord = 1.0 - coord;
    coord *= coord;
    coord = 1.0 - coord;
    vza = zha * coord;
  } else {
    var coord = uv.y * 2.0 - 1.0;
    coord *= coord;
    vza = zha + beta * coord;
  }
  let vzc = cos(vza);
  let vzs = sin(vza);
  let cx = uv.x * uv.x;
  let lvc = -(cx * 2.0 - 1.0);
  let lvs = sqrt(max(0.0, 1.0 - lvc * lvc));
  let rd = vec3f(vzs * lvc, vzc, vzs * lvs);
  let smu = F.sunDir.y;
  let sun = vec3f(sqrt(max(0.0, 1.0 - smu * smu)), smu, 0.0);
  let ro = vec3f(0.0, r, 0.0);

  let tTop = raySphere(ro, rd, RT);
  let tGr = raySphere(ro, rd, RG);
  var tMax = tTop.y;
  if (tGr.x > 0.0) { tMax = tGr.x; }
  let cosT = dot(rd, sun);
  let pR = phaseRayleigh(cosT);
  let pM = phaseMie(cosT, 0.84);
  var L = vec3f(0.0);
  var T = vec3f(1.0);
  let STEPS = 48u;
  var tPrev = 0.0;
  for (var i = 0u; i < STEPS; i++) {
    let fa = (f32(i) + 1.0) / f32(STEPS);
    let tn = tMax * fa * fa;
    let dt = tn - tPrev;
    let t = tPrev + dt * 0.3;
    tPrev = tn;
    let p = ro + rd * t;
    let pr = length(p);
    let h = pr - RG;
    let m = medium(h, haze);
    let pmu = dot(p, sun) / pr;
    var sunT = sampleTrans(pr, pmu);
    if (sunBelowHorizon(pr, pmu)) { sunT = vec3f(0.0); }
    let msuv = vec2f(fromUnitToSubUvs(pmu * 0.5 + 0.5, MS_SIZE), fromUnitToSubUvs(clamp(h / (RT - RG), 0.0, 1.0), MS_SIZE));
    let ms = textureSampleLevel(msLUT, sampC, msuv, 0.0).rgb;
    let S = sunT * (m.rayS * pR + vec3f(m.mieS * pM)) + ms * (m.rayS + vec3f(m.mieS));
    let Ts = exp(-m.ext * dt);
    let ext = max(m.ext, vec3f(1e-9));
    L += T * (S - S * Ts) / ext;
    T *= Ts;
  }
  textureStore(dst, id.xy, vec4f(L, 1.0));
}
`;

// Functions for rendering the sky (background, env cube, ocean fog).
// Requires: F (Frame), transLUT, skyLUT, sampC declared by the including module.
export const SKY_RENDER_FUNCS = NOISE_WGSL + /* wgsl */ `
const SKY_W: f32 = ${SKY_W}.0;
const SKY_H: f32 = ${SKY_H}.0;

fn camR() -> f32 { return RG + F.atmo.y; }

fn sampleTrans(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(transLUT, sampC, transRMuToUV(r, mu), 0.0).rgb;
}

fn sunIlluminance() -> vec3f {
  return sampleTrans(camR(), F.sunDir.y) * F.sunDir.w;
}

fn skyViewUV(dir: vec3f) -> vec2f {
  let r = camR();
  let vHorizon = rhoOf(r);
  let cosBeta = vHorizon / r;
  let beta = acos(clamp(cosBeta, -1.0, 1.0));
  let zha = PI - beta;
  let vzc = clamp(dir.y, -1.0, 1.0);
  var uv: vec2f;
  if (vzc >= -cosBeta) {
    var coord = acos(vzc) / zha;
    coord = 1.0 - coord;
    coord = sqrt(max(coord, 0.0));
    coord = 1.0 - coord;
    uv.y = coord * 0.5;
  } else {
    var coord = (acos(vzc) - zha) / beta;
    coord = sqrt(max(coord, 0.0));
    uv.y = coord * 0.5 + 0.5;
  }
  let dl = length(dir.xz);
  let sl = length(F.sunDir.xz);
  var lvc = 1.0;
  if (dl > 1e-6 && sl > 1e-6) { lvc = clamp(dot(dir.xz, F.sunDir.xz) / (dl * sl), -1.0, 1.0); }
  uv.x = sqrt(max(-lvc * 0.5 + 0.5, 0.0));
  return vec2f(fromUnitToSubUvs(uv.x, SKY_W), fromUnitToSubUvs(uv.y, SKY_H));
}

// Artificial skyglow of a large city: sodium/LED light scattered by the urban haze.
// Brightest at the horizon, warm-grey, independent of the sun.
fn skyGlow(dir: vec3f) -> vec3f {
  let e = max(dir.y, 0.0);
  let band = exp(-e / 0.10) * 0.75 + exp(-e / 0.45) * 0.25;
  // sodium/LED orange near the horizon, a cooler grey-blue overhead (moonlit, less aerosol)
  let tint = mix(vec3f(0.42, 0.48, 0.66), vec3f(1.0, 0.72, 0.48), smoothstep(0.0, 0.6, band));
  return tint * (band * 0.85 + 0.15) * F.night.z;
}
fn skyAtmo(dir: vec3f) -> vec3f {
  return textureSampleLevel(skyLUT, sampC, skyViewUV(dir), 0.0).rgb * F.sunDir.w + skyGlow(dir);
}

// ---- fair-weather cumulus: footprint field (shared by the sky and the scene's cloud shadows) --
const CLOUD_ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);
// fbm whose octaves fade once smaller than the pixel footprint fp (noise units);
// 'billow' turns the small octaves into bulgy |noise| for cauliflower-like edges
fn cloudFbm(p: vec2f, fp: f32, oct: i32, billow: f32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  var f = 1.0;
  for (var i = 0; i < oct; i++) {
    // fade octaves well before they reach pixel size (half-res buffer) -> no shimmer in motion
    let w = 1.0 - smoothstep(0.08, 0.28, fp * f);
    var n = gnoise(q);
    if (i >= 2) { n = mix(n, abs(n) * 2.0 - 0.38, billow); }
    s += a * n * w;
    q = CLOUD_ROT * q * 2.03 + vec2f(13.1, 7.7);
    f *= 2.03;
    a *= 0.5;
  }
  return s;
}
// cloud base and top (km above the river): cumulus humilis / mediocris of an autumn afternoon
fn cuBase() -> f32 { return F.cloudP.x; }
fn cuTop() -> f32 { return F.cloudP.x + 0.85; }
// q: horizontal position (km, world x/z). > 0 under a cloud; larger = thicker, taller core.
fn cuCover(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  // drifting with a ~6 m/s north-westerly
  var p = q * 0.6 + vec2f(-t * 0.0022, -t * 0.0028);
  let w = vec2f(gnoise(p * 0.35 + vec2f(1.7, 9.2)), gnoise(p * 0.35 + vec2f(8.3, 2.8)));
  p += w * 0.8;
  let big = gnoise(q * 0.07 + vec2f(3.1, 1.3)) * 0.4;
  let base = big + 0.5 - (1.0 - F.cloudP.y);
  // octaves summed by hand so clear sky can stop early: after two octaves the remaining ones
  // can add at most ~0.36
  var s = 0.0; var a = 0.5; var f = 1.0; var pp = p;
  for (var i = 0; i < oct; i++) {
    let w = 1.0 - smoothstep(0.08, 0.28, fp * 0.6 * f);
    var nn = gnoise(pp);
    if (i >= 2) { nn = mix(nn, abs(nn) * 2.0 - 0.38, 0.3); }
    s += a * nn * w;
    if (i == 1 && base + s * 1.4 < -0.42) { break; }
    pp = CLOUD_ROT * pp * 2.03 + vec2f(13.1, 7.7);
    f *= 2.03;
    a *= 0.5;
  }
  return base + s * 1.4;
}
// direct-sun transmittance through the cumulus deck above a scene point (baked into the
// cloud shadow map each frame; surfaces read the map)
fn cloudShadow(world: vec3f) -> f32 {
  if (F.cloudP.y < 0.001 || F.sunDir.y < 0.03) { return 1.0; }
  let hm = (cuBase() + cuTop()) * 0.5;
  let q = world.xz * 0.001 + F.sunDir.xz / F.sunDir.y * (hm - world.y * 0.001);
  let c = cuCover(q, 0.04, 3);
  return mix(1.0, 0.04, smoothstep(0.0, 0.12, c));
}
`;

// Clouds + full sky radiance (needs the same bindings as SKY_RENDER_FUNCS).
export const SKY_CLOUD_FUNCS = /* wgsl */ `

// ---- clouds -------------------------------------------------------------
// A volumetric cumulus slab (base ~1.5 km, ray-marched; footprints from cuCover) and a thin
// high layer (4.5 km) shaded as a sheet with self-shadowing towards the sun.

// value noise in [0, 1] for the 3D erosion of cumulus edges
fn vnoise3(p: vec3f) -> f32 {
  let i = vec3i(floor(p));
  let f = p - floor(p);
  let u = f * f * (3.0 - 2.0 * f);
  var v: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    let c = i + o;
    v[k] = f32(pcgHash((u32(c.x) * 73856093u) ^ pcgHash((u32(c.y) * 19349663u) ^ pcgHash(u32(c.z) * 83492791u))) >> 8u) / 16777216.0;
  }
  return mix(mix(mix(v[0], v[1], u.x), mix(v[2], v[3], u.x), u.y), mix(mix(v[4], v[5], u.x), mix(v[6], v[7], u.x), u.y), u.z);
}
// density at (x km, altitude km, z km). Flat bases and domed tops whose height follows the
// footprint strength.
// 'erode' (0..1) cuts the edges into billows; callers fade it out where the march is coarse.
fn cuDensity(pos: vec3f, fp: f32, erode: f32) -> f32 {
  let h = (pos.y - cuBase()) / (cuTop() - cuBase());
  if (h <= 0.0 || h >= 1.0) { return 0.0; }
  let c = cuCover(pos.xz, fp, select(3, 4, erode > 0.0));
  if (c < -0.1) { return 0.0; }
  // iso-surface c * 2.8 = 1.3 h^2.5 around a footprint peak: a flat-based dome
  var d = (c * 2.8 - h * h * sqrt(h) * 1.3) * 1.8;
  d *= smoothstep(0.0, 0.08, h);
  if (erode > 0.0 && d > -0.15) {
    let t = F.cloudP.w;
    let wp = pos * vec3f(7.0, 9.0, 7.0) + vec3f(-t * 0.02, t * 0.004, -t * 0.028);
    let n = vnoise3(wp) * 0.62 + vnoise3(wp * 2.7 + vec3f(5.1, 2.3, 7.7)) * 0.38;
    d -= (n - 0.25) * 0.42 * (1.0 - clamp(d * 1.6, 0.0, 1.0)) * erode;
  }
  return clamp(d, 0.0, 1.0);
}

// Ray-march the slab. Returns (premultiplied radiance, opacity).
fn cumulus(dir: vec3f, pixAng: f32, steps: i32, jitter: f32) -> vec4f {
  if (F.cloudP.y < 0.001 || dir.y <= 0.0) { return vec4f(0.0); }
  let ro = vec3f(0.0, camR(), 0.0);
  let t0 = raySphere(ro, dir, RG + cuBase()).y;
  let t1 = raySphere(ro, dir, RG + cuTop()).y;
  // clear band near the horizon, where the low sun shines under the deck
  let far = 1.0 - smoothstep(22.0, 40.0, t0);
  if (t0 <= 0.0 || far <= 0.0) { return vec4f(0.0); }
  let tEnd = min(t1, t0 + 9.0);
  let camQ = F.camPos.xz * 0.001;
  // 1) coarse search for the first cloud along the ray (no erosion, cheap early-outs)
  let nSearch = 16;
  let dtc = (tEnd - t0) / f32(nSearch);
  var tHit = -1.0;
  for (var i = 0; i < nSearch; i++) {
    let t = t0 + (f32(i) + jitter) * dtc;
    let p = ro + dir * t;
    let r = length(p);
    if (cuDensity(vec3f(p.x + camQ.x, r - RG, p.z + camQ.y), t * pixAng, 0.0) > 0.0) { tHit = max(t - dtc, t0); break; }
  }
  if (tHit < 0.0) { return vec4f(0.0); }
  // 2) fine march from just before the hit
  let dt = clamp((tEnd - tHit) / f32(steps), 0.03, 0.3);
  let sd = F.sunDir.xyz;
  let cosT = dot(dir, sd);
  let sigma0 = F.cloudP.z;   // extinction per km at density 1
  // ambient: sky dome above, horizon band at the sides, a dim ground bounce below
  let skyUp = skyAtmo(vec3f(0.0, 1.0, 0.0));
  let skySide = skyAtmo(normalize(vec3f(dir.x, 0.05, dir.z)));
  var T = 1.0;
  var L = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let t = tHit + (f32(i) + jitter) * dt;
    if (t > tEnd) { break; }
    let p = ro + dir * t;
    let r = length(p);
    let alt = r - RG;
    let pos = vec3f(p.x + camQ.x, alt, p.z + camQ.y);
    let fp = t * pixAng;
    // billowy erosion only where the steps resolve it (~140 m noise wavelength)
    let d = cuDensity(pos, fp, 1.0 - smoothstep(0.12, 0.3, dt));
    if (d > 0.002) {
      let up = p / r;
      let smu = dot(up, sd);
      let sunC = sampleTrans(r, smu) * F.sunDir.w * select(1.0, 0.0, sunBelowHorizon(r, smu));
      // optical depth towards the sun (2 taps out to ~0.6 km)
      let tauS = (cuDensity(pos + sd * 0.08, fp, 0.0) * 0.18 + cuDensity(pos + sd * 0.35, fp, 0.0) * 0.45) * sigma0;
      // multiple scattering as a sum of ever softer, less extinguished octaves
      var ms = 0.0;
      var a = 1.0; var b = 1.0; var c = 1.0;
      for (var o = 0; o < 3; o++) {
        ms += a * exp(-tauS * b) * mix(phaseHG(cosT, 0.75 * c), phaseHG(cosT, -0.25 * c), 0.3);
        a *= 0.5; b *= 0.45; c *= 0.5;
      }
      // dark edges facing the viewer when front-lit ("powder"): little in-scattering at the
      // thin rim, fading when looking towards the sun
      let powder = mix(1.0 - exp(-d * sigma0 * 0.18), 1.0, smoothstep(-0.1, 0.8, cosT));
      let h = clamp((alt - cuBase()) / (cuTop() - cuBase()), 0.0, 1.0);
      let amb = (skyUp * (0.3 + 0.7 * h) + skySide * 0.3) * (1.0 - 0.5 * d);
      // at night the bases reflect the city (the orange ceiling); moonlight on the tops
      let night = vec3f(1.0, 0.70, 0.45) * F.night.z * 0.9 * (1.0 - 0.6 * h)
                + F.moonDir.w * max(F.moonDir.y, 0.0) * vec3f(0.62, 0.68, 0.80) * 0.3;
      let S = sunC * ms * powder * 3.0 + amb + night;
      let Tr = exp(-d * sigma0 * dt);
      L += T * S * (1.0 - Tr);
      T *= Tr;
      if (T < 0.02) { break; }
    }
  }
  let alpha = (1.0 - T) * far;
  // aerial perspective between the camera and the deck
  let vis = exp(-t0 * 0.02 * (0.6 + 0.4 * F.atmo.x));
  return vec4f(L * far * vis, alpha * vis);
}

fn highEdge(q: vec2f, fp: f32, oct: i32) -> f32 {
  let t = F.cloudP.w;
  // altostratus deck: smooth, slightly fibrous
  let p = CLOUD_ROT * (q * vec2f(0.09, 0.20)) + vec2f(t * 0.0015, 0.0);
  let n = cloudFbm(p, fp * 0.2, oct, 0.0) * 1.25 + 0.5;
  let big = gnoise(q * 0.012 + vec2f(7.0, 3.0)) * 0.6;
  return n + big - (1.0 - F.atmo.z);
}
fn highSoft(fp: f32) -> f32 { return 0.85 + min(fp * 0.2, 1.0) * 0.3; }
fn highDensity(q: vec2f, fp: f32, oct: i32) -> f32 {
  return clamp(highEdge(q, fp, oct) / highSoft(fp), 0.0, 1.0);
}

struct CloudHit { t: f32, q: vec2f, mu: f32, up: vec3f };
fn hitLayer(dir: vec3f, hc: f32) -> CloudHit {
  var h: CloudHit;
  let ro = vec3f(0.0, camR(), 0.0);
  h.t = raySphere(ro, dir, RG + hc).y;
  let p = ro + dir * h.t;
  h.up = p / length(p);
  h.q = p.xz + F.camPos.xz * 0.001;
  h.mu = max(dot(h.up, dir), 0.03);
  return h;
}

// d: density at the hit; s1..s3: densities 80 m / 250 m / 600 m towards the sun
fn shadeLayer(dir: vec3f, h: CloudHit, hc: f32, d: f32, s1: f32, s2: f32, s3: f32, sigma: f32, g: f32) -> vec4f {
  let tau = d * sigma / h.mu;
  let alpha = 1.0 - exp(-tau);
  let tauS = (d * 0.5 + s1 * 0.6 + s2 + s3 * 0.7) * sigma * 0.55;
  let Ts = exp(-tauS);
  let smu = dot(h.up, F.sunDir.xyz);
  // no direct sun once it has set for the cloud layer too
  let sunC = sampleTrans(RG + hc, smu) * F.sunDir.w * select(1.0, 0.0, sunBelowHorizon(RG + hc, smu));
  let cosT = dot(dir, F.sunDir.xyz);
  let ph = mix(phaseHG(cosT, g), phaseHG(cosT, -0.2), 0.25);
  let fwd = smoothstep(-0.2, 0.7, cosT);
  // single scattering: reflected when front-lit, transmitted through thin parts when back-lit
  let single = sunC * Ts * ph * mix(alpha * 0.55, tau * exp(-tau) * 2.2, fwd) * 3.0;
  // edges where density falls off towards the sun catch direct light
  let rim = clamp((d - s1) * 2.5, 0.0, 1.0);
  // multiple scattering: a low sun lights cloud tops and sun-facing flanks, not the bases we
  // see from below, so diffused sunlight shows mostly when looking towards the sun
  let toward = smoothstep(-0.3, 0.9, cosT);
  let ms = sunC * (0.025 + 0.22 * rim) * mix(1.0, Ts, 0.5) * alpha * (0.25 + 0.75 * toward);
  // bases are lit by the sky from the sides/below (horizon band + sea), cooler than the tops
  let amb = (skyAtmo(normalize(vec3f(dir.x, 0.03, dir.z))) * 0.26 + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 0.42)
            * alpha * (1.0 - 0.35 * d);
  // at night the cloud bases reflect the city's light back down (the familiar orange ceiling)
  let cityLit = vec3f(1.0, 0.70, 0.45) * F.night.z * 2.2 * alpha * (0.6 + 0.4 * d);
  // moonlight on the cloud deck
  let moonC = F.moonDir.w * max(F.moonDir.y, 0.0) * vec3f(0.62, 0.68, 0.80) * alpha * 0.35;
  return vec4f(single + ms + amb + cityLit + moonC, alpha);
}

// returns (premultiplied radiance, fraction of the background sky hidden)
// aa: per-layer edge change per pixel (fwidth of the edge function); negative = estimate it
// from the pixel footprint (compute passes, which have no derivatives)
// lo: the cumulus in front (premultiplied radiance, opacity), normally read from the cached
// dome (cloudDomeAt); zero for the sun visibility pass (the scene has per-point cloud shadows)
fn cloudLayers(dir: vec3f, pixAng: f32, aa: vec2f, lo: vec4f, sunT: ptr<function, f32>) -> vec4f {
  *sunT = 1.0;
  if (dir.y <= 0.0) { return vec4f(0.0); }
  let sxz = F.sunDir.xz / max(length(F.sunDir.xz), 1e-4);
  var hi = vec4f(0.0);
  var hiVis = 1.0;
  if (F.atmo.z > 0.001) {
    let hc = 4.5;
    let h = hitLayer(dir, hc);
    let far = 1.0 - smoothstep(30.0, 70.0, h.t);
    if (far > 0.0) {
      let fp = h.t * pixAng / h.mu;
      let e = highEdge(h.q, fp, 6);
      let d = clamp(e / highSoft(fp), 0.0, 1.0) * far;
      if (d > 0.002) {
        let s1 = highDensity(h.q + sxz * 0.15, fp, 4) * far;
        let s2 = highDensity(h.q + sxz * 0.5, fp, 3) * far;
        hi = shadeLayer(dir, h, hc, d, s1, s2, s2, F.atmo.w, 0.80);
        // drop the faint fringe of the deck: under strong forward scattering near a low sun it
        // otherwise shows as concentric contour bands across the clear sky
        hi *= smoothstep(0.02, 0.12, d);
        let aw = select(fp * 0.2 * 2.5, aa.y, aa.y >= 0.0);
        hi *= smoothstep(0.0, aw * 1.5 + 1e-6, e);
        hiVis = exp(-h.t * 0.010 * (0.6 + 0.4 * F.atmo.x));
      }
    }
  }
  *sunT = (1.0 - lo.a) * (1.0 - hi.a);
  // low layer in front; aerial perspective fades each layer into the haze without making
  // it transparent to the layer behind
  let col = lo.rgb + hi.rgb * hiVis * (1.0 - lo.a);
  let hidden = lo.a + hi.a * hiVis * (1.0 - lo.a);
  return vec4f(col, hidden);
}

fn sunDisk(dir: vec3f) -> vec3f {
  let c = dot(dir, F.sunDir.xyz);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(SUN_ANG_R - pix, SUN_ANG_R + pix, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  let x = min(ang / SUN_ANG_R, 1.0);
  let mu = sqrt(max(1.0 - x * x, 0.0));
  let limb = vec3f(1.0) - vec3f(0.397, 0.503, 0.652) * (1.0 - mu);
  let rad = F.sunDir.w / (PI * SUN_ANG_R * SUN_ANG_R);
  return rad * limb * edge * sampleTrans(camR(), max(dir.y, -0.01));
}

const MOON_ANG_R: f32 = 0.0045;
fn moonDisk(dir: vec3f) -> vec3f {
  if (F.moonDir.w <= 0.0 || F.moonDir.y < -0.02) { return vec3f(0.0); }
  let c = dot(dir, F.moonDir.xyz);
  let ang = sqrt(max(2.0 * (1.0 - c), 0.0));
  let pix = F.screen.w * 2.0 * length(F.camUp.xyz) * 0.75;
  let edge = 1.0 - smoothstep(MOON_ANG_R - pix, MOON_ANG_R + pix, ang);
  if (edge <= 0.0) { return vec3f(0.0); }
  // faint maria texture
  let q = (dir - F.moonDir.xyz * c) / MOON_ANG_R;
  let maria = 0.82 + 0.18 * gnoise(q.xy * 2.3 + q.z * 1.7);
  // disc radiance = illuminance / solid angle (albedo already folded into the illuminance)
  let rad = F.moonDir.w / (PI * MOON_ANG_R * MOON_ANG_R);
  return vec3f(1.0, 0.97, 0.92) * rad * edge * maria * sampleTrans(camR(), max(dir.y, 0.0));
}

fn skyRadiance(dirIn: vec3f, withSun: bool, pixAng: f32, lo: vec4f) -> vec3f {
  var dir = dirIn;
  dir.y = max(dir.y, 0.0005);
  dir = normalize(dir);
  var L = skyAtmo(dir);
  var sunT = 1.0;
  let cl = cloudLayers(dir, pixAng, vec2f(-1.0), lo, &sunT);
  L = L * (1.0 - cl.a) + cl.rgb;
  if (withSun) { L += sunDisk(dir) * sunT; }
  return L;
}
`;

// Cached cumulus panorama: u = azimuth, v = sqrt(elevation / 90 deg) (more rows near the
// horizon, where clouds are small and dense). Requires a 'cloudDome' texture binding.
export const CLOUD_DOME_SAMPLE = /* wgsl */ `
const DOME_W: f32 = ${DOME_W}.0;
const DOME_H: f32 = ${DOME_H}.0;
fn domeDir(uv: vec2f) -> vec3f {
  let az = (uv.x - 0.5) * 2.0 * PI;
  let el = uv.y * uv.y * 0.5 * PI;
  return vec3f(sin(az) * cos(el), sin(el), -cos(az) * cos(el));
}
fn cloudDomeAt(dir: vec3f, smp: sampler) -> vec4f {
  if (dir.y <= 0.0) { return vec4f(0.0); }
  let az = atan2(dir.x, -dir.z);
  let el = asin(clamp(dir.y, 0.0, 1.0));
  // keep v inside the first/last texel centres: a repeating sampler would wrap it
  let v = clamp(sqrt(el / (0.5 * PI)), 0.5 / DOME_H, 1.0 - 0.5 / DOME_H);
  // a small tent filter hides the per-texel jitter grain of the march
  let uv = vec2f(az / (2.0 * PI) + 0.5, v);
  let o = vec2f(0.6 / DOME_W, 0.6 / DOME_H);
  return (textureSampleLevel(cloudDome, smp, uv + vec2f(o.x, o.y), 0.0) + textureSampleLevel(cloudDome, smp, uv + vec2f(-o.x, o.y), 0.0)
        + textureSampleLevel(cloudDome, smp, uv + vec2f(o.x, -o.y), 0.0) + textureSampleLevel(cloudDome, smp, uv - o, 0.0)) * 0.25;
}
`;

// Ray-marches the cumulus into the dome: every frame one row in thirty-two (z = extra phases on
// the first frames, so the whole dome fills at once).
export const CLOUD_DOME = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
const DOME_W: f32 = ${DOME_W}.0;
const DOME_H: f32 = ${DOME_H}.0;
fn domeDir(uv: vec2f) -> vec3f {
  let az = (uv.x - 0.5) * 2.0 * PI;
  let el = uv.y * uv.y * 0.5 * PI;
  return vec3f(sin(az) * cos(el), sin(el), -cos(az) * cos(el));
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.y * 32u + (u32(F.cloudMap.w) + id.z) % 32u;
  if (f32(id.x) >= DOME_W || f32(row) >= DOME_H) { return; }
  let uv = (vec2f(f32(id.x), f32(row)) + 0.5) / vec2f(DOME_W, DOME_H);
  let dir = domeDir(uv);
  // angular texel size sets the noise octave cut-off; a per-texel jitter de-bands the march
  let pixAng = 2.0 * PI / DOME_W;
  let jit = f32(pcgHash(id.x * 7919u + row * 104729u) >> 8u) / 16777216.0;
  textureStore(dst, vec2u(id.x, row), cumulus(dir, pixAng, 18, jit));
}
`;

// Cumulus shadow transmittance on the ground around the camera (r channel).
export const CLOUD_SHADOW = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
` + SKY_RENDER_FUNCS + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = ${CSHADOW_SIZE}u;
  if (id.x >= n || id.y >= n) { return; }
  let xz = F.cloudMap.xy + (vec2f(id.xy) + 0.5) / f32(n) * F.cloudMap.z;
  textureStore(dst, id.xy, vec4f(cloudShadow(vec3f(xz.x, 0.0, xz.y)), 0.0, 0.0, 1.0));
}
`;

export const ENV_CUBE = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var cloudDome: texture_2d<f32>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + CLOUD_DOME_SAMPLE + /* wgsl */ `
const ENV: f32 = ${ENV_SIZE}.0;
fn faceDir(face: u32, uv: vec2f) -> vec3f {
  let u = uv.x * 2.0 - 1.0;
  let v = uv.y * 2.0 - 1.0;
  switch (face) {
    case 0u: { return vec3f(1.0, -v, -u); }
    case 1u: { return vec3f(-1.0, -v, u); }
    case 2u: { return vec3f(u, 1.0, v); }
    case 3u: { return vec3f(u, -1.0, -v); }
    case 4u: { return vec3f(u, -v, 1.0); }
    default: { return vec3f(-u, -v, -1.0); }
  }
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (f32(id.x) >= ENV || f32(id.y) >= ENV) { return; }
  // faces are refreshed round-robin: face = (frame + z) % 6
  let face = (u32(F.post.z) + id.z) % 6u;
  let dir = normalize(faceDir(face, (vec2f(id.xy) + 0.5) / ENV));
  var L: vec3f;
  if (dir.y >= 0.0) {
    L = skyRadiance(dir, false, 1.5708 / ENV * 1.5, cloudDomeAt(dir, sampC));
  } else {
    // below the horizon: hazy distant ground / water lit by the sky and the sun
    let hz = skyRadiance(normalize(vec3f(dir.x, 0.004, dir.z)), false, 1.5708 / ENV * 1.5, vec4f(0.0));
    let ground = vec3f(0.07, 0.07, 0.065) * (sunIlluminance() * max(F.sunDir.y, 0.0) + skyAtmo(vec3f(0.0, 1.0, 0.0)) * 2.0);
    L = mix(hz * 0.8, ground, clamp(-dir.y * 5.0, 0.0, 1.0));
  }
  textureStore(dst, id.xy, face, vec4f(L, 1.0));
}
`;

export const CUBE_MIPS = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let srcSize = vec2i(textureDimensions(src));
  let p = vec2i(id.xy) * 2;
  // 4x4 tent-ish filter for smoother roughness blur
  var acc = vec4f(0.0);
  var wsum = 0.0;
  for (var j = -1; j <= 2; j++) {
    for (var i = -1; i <= 2; i++) {
      let q = clamp(p + vec2i(i, j), vec2i(0), srcSize - 1);
      let w = select(1.0, 3.0, i == 0 || i == 1) * select(1.0, 3.0, j == 0 || j == 1);
      acc += textureLoad(src, q, id.z, 0) * w;
      wsum += w;
    }
  }
  textureStore(dst, id.xy, id.z, acc / wsum);
}
`;

// High-layer transmittance towards the sun (averaged over the solar disc), used to dim direct
// sunlight everywhere; the cumulus deck shadows the scene per point (cloudShadow).
export const SUN_VIS = FRAME_WGSL + ATMO_CORE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var<storage, read_write> sunVis: array<f32, 4>;
` + SKY_RENDER_FUNCS + SKY_CLOUD_FUNCS + /* wgsl */ `
var<workgroup> part: array<f32, 8>;
// one ray per thread (7 rays across the disc), then a tiny reduction
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_index) k: u32) {
  let sd = F.sunDir.xyz;
  let tx = normalize(cross(sd, vec3f(0.0, 1.0, 0.0)));
  let ty = cross(tx, sd);
  var t = 1.0;
  if (k < 7u) {
    let a = f32(k) * 0.8975979;
    let o = select(vec2f(cos(a), sin(a)) * 0.65, vec2f(0.0), k == 0u);
    let dir = normalize(sd + (tx * o.x + ty * o.y) * SUN_ANG_R);
    let cl = cloudLayers(dir, 0.0, vec2f(-1.0), vec4f(0.0), &t);
  }
  part[k] = t;
  workgroupBarrier();
  if (k == 0u) {
    var acc = 0.0;
    for (var i = 0u; i < 7u; i++) { acc += part[i]; }
    sunVis[0] = acc / 7.0;
  }
}
`;
