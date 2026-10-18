// Shared scene lighting: sun + sky + city light, cascaded shadows, aerial perspective.
// Every lit pipeline (terrain, buildings, bridges, boat, trees, water) uses the same
// functions and bind group 0, so all surfaces sit in one lighting environment.
import { FRAME_WGSL, ATMO_CORE_WGSL, ENV_MIPS, ENV_SIZE } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';

export const SCENE_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var envCube: texture_cube<f32>;
@group(0) @binding(5) var shadowMap: texture_depth_2d_array;
@group(0) @binding(6) var shadowSamp: sampler_comparison;
@group(0) @binding(7) var<storage, read> sunVis: array<f32, 4>;
@group(0) @binding(8) var sampR: sampler;
@group(0) @binding(9) var occTex: texture_2d<f32>;
@group(0) @binding(10) var sceneSdf: texture_2d<f32>;
`;

export const LIGHTING_WGSL = FRAME_WGSL + ATMO_CORE_WGSL + SCENE_BINDINGS + SKY_RENDER_FUNCS + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;
const ENV_TEXELS_PER_RAD: f32 = ${ENV_SIZE}.0 / 1.5708;

fn isReflPass() -> bool { return F.camFwd.w > 0.5; }

// Direct sunlight reaching the scene (atmosphere + cloud transmittance)
fn sunLight() -> vec3f { return sunIlluminance() * sunVis[0] * step(-0.02, F.sunDir.y); }
// Moonlight: moonDir.w is the lunar illuminance in the same units as the sun (full moon is
// ~2.5e-6 of full sunlight); slightly blue-shifted the way night photographs render it.
fn moonLight() -> vec3f {
  return vec3f(0.62, 0.68, 0.80) * F.moonDir.w * smoothstep(-0.02, 0.15, F.moonDir.y);
}

// ---- cascaded shadows ----
fn shadowCascade(rel: vec3f, c: i32, nrm: vec3f) -> f32 {
  var m: mat4x4f;
  if (c == 0) { m = F.shadowVP0; } else if (c == 1) { m = F.shadowVP1; } else { m = F.shadowVP2; }
  let texel = select(select(0.10, 0.6, c == 1), 3.0, c == 2) * 1.5;
  let p = m * vec4f(rel + nrm * texel, 1.0);
  let uv = p.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0 || p.z < 0.0) { return 1.0; }
  let ts = F.shadowP.x;
  var s = 0.0;
  // 3x3 PCF with hardware comparison (bilinear) taps
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(i), f32(j)) * ts * 1.2, c, p.z - 0.00015);
    }
  }
  return s / 9.0;
}
fn shadowAt(rel: vec3f, nrm: vec3f) -> f32 {
  if (F.misc.y < 0.5 || F.sunDir.y < -0.01) { return 1.0; }
  let d = length(rel);
  if (d < F.shadowP.y * 0.92) {
    let s0 = shadowCascade(rel, 0, nrm);
    let k = smoothstep(F.shadowP.y * 0.75, F.shadowP.y * 0.92, d);
    if (k > 0.0) { return mix(s0, shadowCascade(rel, 1, nrm), k); }
    return s0;
  }
  if (d < F.shadowP.z * 0.92) {
    let s1 = shadowCascade(rel, 1, nrm);
    let k = smoothstep(F.shadowP.z * 0.75, F.shadowP.z * 0.92, d);
    if (k > 0.0) { return mix(s1, shadowCascade(rel, 2, nrm), k); }
    return s1;
  }
  if (d < F.shadowP.w) { return mix(shadowCascade(rel, 2, nrm), 1.0, smoothstep(F.shadowP.w * 0.8, F.shadowP.w, d)); }
  return 1.0;
}

// ---- sky occlusion from overhead structures (bridges): a top-down map baked at load ----
// r = occluder underside height above the water (m, 0 = open sky), g = blurred coverage
fn overheadOcclusion(world: vec3f) -> f32 {
  let uv = (world.xz - F.sdfP.xy) / F.sdfP.zw;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  let o = textureSampleLevel(occTex, sampC, uv, 0.0);
  let under = step(world.y + 0.3, o.r) * step(0.01, o.r);
  return 1.0 - max(o.g * 0.85, under * 0.8);
}

// Diffuse sky irradiance / PI (average radiance around the normal) from the env cube
fn skyDiffuse(n: vec3f) -> vec3f {
  let up = textureSampleLevel(envCube, sampC, n, ENV_MAXMIP).rgb;
  // ground bounce for downward-facing surfaces
  let g = vec3f(0.11, 0.105, 0.095) * (sunLight() * max(F.sunDir.y, 0.0) * 0.32 + textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb);
  return mix(g, up, smoothstep(-0.6, 0.4, n.y));
}
// Diffuse light from the city itself at night (street lamps, windows), warm and weak
// Street lamps light the lowest few storeys; above that only a faint glow from the city remains.
fn cityAmbient(n: vec3f, world: vec3f) -> vec3f {
  // street lighting reaches structures on and near the banks, not the middle of the river
  let uv = (world.xz - F.sdfP.xy) / F.sdfP.zw;
  var dBank = 100.0;
  if (all(uv > vec2f(0.0)) && all(uv < vec2f(1.0))) { dBank = textureSampleLevel(sceneSdf, sampC, uv, 0.0).r; }
  let low = (1.0 - smoothstep(3.0, 25.0, world.y)) * smoothstep(-60.0, 5.0, dBank);
  return vec3f(1.0, 0.76, 0.52) * F.night.y * (0.00006 + 0.0005 * low) * (0.6 + 0.4 * max(-n.y, 0.0) + 0.3 * (1.0 - abs(n.y)));
}

fn fresnelSchlick(c: f32, f0: vec3f) -> vec3f {
  let x = 1.0 - clamp(c, 0.0, 1.0);
  let x2 = x * x;
  return f0 + (vec3f(1.0) - f0) * (x2 * x2 * x);
}
// GGX specular for the sun (with the solar disc widening the lobe slightly)
fn ggxSpec(n: vec3f, v: vec3f, l: vec3f, rough: f32, f0: vec3f) -> vec3f {
  let a = max(rough * rough, 0.002);
  let h = normalize(v + l);
  let nh = max(dot(n, h), 0.0);
  let nv = max(dot(n, v), 1e-3);
  let nl = max(dot(n, l), 0.0);
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let k = a * 0.5;
  let G = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
  return fresnelSchlick(dot(v, h), f0) * D * G / (4.0 * nv);
}
// Env-cube specular with roughness-selected mip
fn envSpec(r: vec3f, rough: f32) -> vec3f {
  var rr = r;
  // reflections below the horizon see the dim ground band of the cube
  let lod = clamp(rough * ENV_MAXMIP * 1.1, 0.0, ENV_MAXMIP);
  return textureSampleLevel(envCube, sampC, rr, lod).rgb;
}

// Standard opaque shading used by meshes and terrain.
struct Surf { albedo: vec3f, rough: f32, metal: f32, f0: vec3f, emit: vec3f, ao: f32 };
fn shade(s: Surf, n: vec3f, rel: vec3f, world: vec3f) -> vec3f {
  let dist = length(rel);
  let v = -rel / max(dist, 1e-4);
  let l = F.sunDir.xyz;
  let sh = shadowAt(rel, n);
  let occ = overheadOcclusion(world);
  let sun = sunLight() * sh * occ;
  let nl = max(dot(n, l), 0.0);
  let diffCol = s.albedo * (1.0 - s.metal);
  let f0 = mix(s.f0, s.albedo, s.metal);
  var col = diffCol * (sun * nl / PI);
  col += ggxSpec(n, v, l, s.rough, f0) * sun * nl;
  // moon as a second (weak) directional light, unshadowed
  col += diffCol * moonLight() * max(dot(n, F.moonDir.xyz), 0.0) / PI;
  let amb = skyDiffuse(n) * F.misc.w * occ * s.ao;
  col += diffCol * (amb + cityAmbient(n, world) * s.ao);
  let r = reflect(-v, n);
  let fr = fresnelSchlick(max(dot(n, v), 0.0), f0) * (1.0 - s.rough * 0.7);
  col += envSpec(r, s.rough) * fr * s.ao * occ * smoothstep(-0.35, 0.1, r.y);
  return col + s.emit;
}

// Aerial perspective: extinction plus in-scattered sky light along the view ray.
fn applyFog(col: vec3f, rel: vec3f) -> vec3f {
  let dist = length(rel);
  let v = rel / max(dist, 1e-4);
  // haze is denser near the ground: integrate an exponential height falloff
  let h0 = F.camPos.y;
  let H = 700.0;
  let dy = v.y * dist;
  // optical depth relative to ground-level density: integral of exp(-y/H) along the ray
  var od = dist * exp(-h0 / H);
  if (abs(dy) > 0.5) { od = H * dist / dy * (exp(-h0 / H) - exp(-(h0 + dy) / H)); }
  od = max(od, 0.0);
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * od);
  let hz = normalize(vec3f(v.x, max(v.y, 0.0) * 0.5 + 0.004, v.z));
  let fogCol = skyAtmo(hz);
  return col * T + fogCol * (vec3f(1.0) - T);
}

// Box-filtered square wave (fraction "duty" high) for anti-aliased procedural grids.
fn filteredPulse(x: f32, duty: f32, w: f32) -> f32 {
  let ww = max(w, 1e-4);
  let a = x - 0.5 * ww;
  let b = x + 0.5 * ww;
  let ia = floor(a) * duty + min(fract(a), duty);
  let ib = floor(b) * duty + min(fract(b), duty);
  return (ib - ia) / ww;
}
`;
