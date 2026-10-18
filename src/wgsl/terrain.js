// Land surface: CDLOD grid displaced by the land height map. Near the river the height comes
// from the bank profile evaluated on the (bilinearly interpolated) river SDF, which keeps the
// waterline smooth and sharp at any distance. Materials come from the land class map.
import { LIGHTING_WGSL } from './lighting.js';
import { NOISE_WGSL } from './common.js';

export const BANK_PROFILE_WGSL = /* wgsl */ `
// Mirrors bankProfile() in tools/build-scene.mjs
fn bankProfile(d: f32) -> f32 {
  if (d < 0.0) { return max(-6.0, d * 0.45 - 0.25); }
  if (d < 5.0) { return -0.25 + d / 5.0 * 2.85; }
  if (d < 14.0) { return 2.6 + (d - 5.0) * 0.01; }
  if (d < 40.0) { let t = (d - 14.0) / 26.0; return 2.7 + t * t * 1.3; }
  return 4.0;
}
`;

export const TERRAIN = LIGHTING_WGSL + BANK_PROFILE_WGSL + /* wgsl */ `
@group(1) @binding(0) var sdfTex: texture_2d<f32>;
@group(1) @binding(1) var hgtTex: texture_2d<f32>;
@group(1) @binding(2) var farTex: texture_2d<f32>;
@group(1) @binding(3) var clsTex: texture_2d<u32>;
@group(1) @binding(4) var detailTex: texture_2d<f32>;

fn sdfAt(xz: vec2f) -> f32 {
  let uv = (xz - F.sdfP.xy) / F.sdfP.zw;
  if (any(uv < vec2f(0.002)) || any(uv > vec2f(0.998))) { return 1000.0; }
  return textureSampleLevel(sdfTex, sampC, uv, 0.0).r;
}
fn landHeight(xz: vec2f) -> f32 {
  let d = sdfAt(xz);
  if (d < 40.0) { return bankProfile(d); }
  let uv = (xz - F.hgtP.xy) / F.hgtP.zw;
  if (all(uv > vec2f(0.001)) && all(uv < vec2f(0.999))) {
    let h = textureSampleLevel(hgtTex, sampC, uv, 0.0).r;
    // join the SDF-driven bank smoothly
    return mix(bankProfile(d), h, smoothstep(40.0, 70.0, d));
  }
  let fuv = (xz - F.farP.xy) / F.farP.zw;
  let hf = textureSampleLevel(farTex, sampC, fuv, 0.0).r;
  // blend into the far terrain at the edge of the detailed map
  let e = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
  let hn = textureSampleLevel(hgtTex, sampC, clamp(uv, vec2f(0.001), vec2f(0.999)), 0.0).r;
  return mix(hf, hn, smoothstep(-0.05, 0.0, e));
}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) world: vec3f,
  @location(2) nrm: vec3f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> VSOut {
  let n = u32(F.lodP.y);
  let g = vec2f(f32(vi % (n + 1u)), f32(vi / (n + 1u)));
  let spacing = inst.z / f32(n);
  var xz = inst.xy + g * spacing;
  let R = F.lodP.z * exp2(inst.w);
  let d = distance(vec3f(xz.x, landHeight(xz), xz.y), F.camPos.xyz);
  let mk = clamp((d - 0.8 * R) / (0.2 * R), 0.0, 1.0);
  xz -= fract(g * 0.5) * 2.0 * spacing * mk;
  var o: VSOut;
  let world = vec3f(xz.x, landHeight(xz), xz.y);
  o.world = world;
  o.rel = world - F.camPos.xyz;
  // normal from central differences at the local grid spacing (smooth across the mesh)
  let e = max(spacing * (1.0 + mk), 0.5);
  let hx = landHeight(xz + vec2f(e, 0.0)) - landHeight(xz - vec2f(e, 0.0));
  let hz = landHeight(xz + vec2f(0.0, e)) - landHeight(xz - vec2f(0.0, e));
  o.nrm = normalize(vec3f(-hx, 2.0 * e, -hz));
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

fn landClass(xz: vec2f) -> u32 {
  let uv = (xz - F.sdfP.xy) / F.sdfP.zw;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 0u; }
  let dims = vec2f(textureDimensions(clsTex));
  return textureLoad(clsTex, vec2i(uv * dims), 0).r;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (isReflPass() && in.world.y < -0.3) { discard; }
  let xz = in.world.xz;
  let n = normalize(in.nrm);
  let d = sdfAt(xz);
  // stair coordinate for stepped revetments (derivative taken in uniform control flow)
  let stepK = in.world.y / 0.4;
  let fwStep = fwidth(stepK);
  let cls = landClass(xz);

  // ---- material ----
  let det = textureSample(detailTex, sampR, xz * 0.21).r;   // fine detail (1/0.21 m tiles)
  let det2 = textureSample(detailTex, sampR, xz * 0.031).b;  // broad variation
  let rock = textureSample(detailTex, sampR, xz * 0.55).g;
  // mip level of the stone texture, for lookups inside the (non-uniform) revetment branch
  let rockLod = log2(max(length(fwidth(xz * 0.55)) * 512.0, 1.0));
  var alb = vec3f(0.30, 0.29, 0.27);   // urban default (paving, roofs below the building mesh)
  var rough = 0.85;
  if (cls == 1u) { alb = mix(vec3f(0.11, 0.17, 0.06), vec3f(0.19, 0.21, 0.09), det2) * (0.85 + 0.3 * det); rough = 0.95; }
  else if (cls == 2u) { alb = vec3f(0.06, 0.10, 0.045) * (0.8 + 0.4 * det); rough = 1.0; }
  else if (cls == 3u) { alb = vec3f(0.085, 0.085, 0.09) * (0.9 + 0.2 * det); rough = 0.75; }
  else if (cls == 4u) { alb = vec3f(0.45, 0.40, 0.32) * (0.9 + 0.2 * det); }
  else if (cls == 5u) { alb = vec3f(0.22, 0.19, 0.17) * (0.8 + 0.3 * det); }
  else if (cls == 6u) { alb = vec3f(0.38, 0.30, 0.26) * (0.9 + 0.15 * det); rough = 0.8; } // red cycle path
  else if (cls == 7u) { alb = vec3f(0.36, 0.35, 0.33) * (0.88 + 0.2 * det); rough = 0.8; }
  // Revetment (the 30-degree band between the water and the promenade). Two kinds alternate
  // along the river in stretches of a few hundred metres: grey riprap with weeds between the
  // stones, and stepped concrete terraces (0.4 m rise, 0.7 m tread - the same slope). Which
  // stretch gets which is procedural, not mapped.
  var nn = n;
  if (d < 5.5 && d > -6.0) {
    let band = 1.0 - smoothstep(4.6, 5.4, d);
    let stepped = smoothstep(0.52, 0.56, textureSampleLevel(detailTex, sampR, xz * 0.0011, 0.0).b);
    // riprap: dark grey stones, bumpy, weeds and dry grass in the joints higher up
    let rg = vec2f(textureSampleLevel(detailTex, sampR, (xz + vec2f(0.25, 0.0)) * 0.55, rockLod).g - rock,
                   textureSampleLevel(detailTex, sampR, (xz + vec2f(0.0, 0.25)) * 0.55, rockLod).g - rock);
    var stone = vec3f(0.20, 0.195, 0.18) * (0.5 + 0.7 * rock);
    let weeds = smoothstep(0.55, 0.75, det2 + 0.25 * det) * smoothstep(0.8, 1.6, in.world.y) * (1.0 - rock * 0.8);
    stone = mix(stone, mix(vec3f(0.07, 0.10, 0.04), vec3f(0.20, 0.18, 0.10), det), weeds);
    let nRip = normalize(n + vec3f(-rg.x, 0.0, -rg.y) * 1.6 * (1.0 - stepped));
    // steps: treads face up, risers face the water
    let riser = 1.0 - filteredPulse(stepK, 0.64, fwStep);
    let hd = normalize(n.xz + vec2f(1e-4, 0.0));
    let nStep = normalize(mix(vec3f(0.0, 1.0, 0.0), vec3f(hd.x, 0.15, hd.y), riser));
    let conc = vec3f(0.29, 0.285, 0.27) * (0.85 + 0.25 * det) * mix(1.0, 0.8, riser) * (0.9 + 0.2 * det2);
    // flood silt and moss on the lowest steps, patchy grime higher up
    let moss = (1.0 - smoothstep(0.6, 1.5, in.world.y)) * (0.6 + 0.4 * det2);
    let conc2 = mix(conc * mix(1.0, 0.7, smoothstep(0.55, 0.8, det2)), vec3f(0.10, 0.11, 0.07), moss);
    let rev = mix(stone, conc2, stepped);
    alb = mix(alb, rev, band);
    rough = mix(rough, mix(0.75, 0.85, stepped), band);
    nn = normalize(mix(n, mix(nRip, nStep, stepped), band));
    // concrete kerb along the top edge of the revetment
    let kerb = smoothstep(4.55, 4.7, d) * (1.0 - smoothstep(5.05, 5.2, d));
    alb = mix(alb, vec3f(0.46, 0.45, 0.42) * (0.9 + 0.15 * det), kerb);
    // wet/stained band just above the water (water level fluctuates ~0.5 m)
    let wet = 1.0 - smoothstep(0.0, 0.55, in.world.y);
    alb *= mix(1.0, 0.45, wet);
    rough = mix(rough, 0.3, wet);
    // algae tint just at the waterline
    alb = mix(alb, vec3f(0.08, 0.10, 0.05), (1.0 - smoothstep(0.0, 0.18, abs(in.world.y - 0.05))) * 0.6);
  }
  // mountains far away: forested
  if (d >= 1000.0 && in.world.y > 40.0) { alb = mix(alb, vec3f(0.06, 0.09, 0.05), smoothstep(40.0, 90.0, in.world.y)); }

  var s: Surf;
  s.albedo = alb; s.rough = rough; s.metal = 0.0; s.f0 = vec3f(0.035); s.emit = vec3f(0.0); s.ao = 1.0;
  // street/park light pools on paths and roads at night
  if (cls == 3u || cls == 6u || cls == 7u) { s.emit = alb * vec3f(1.0, 0.75, 0.5) * F.night.y * 0.006; }
  var col = shade(s, nn, in.rel, in.world);
  col = applyFog(col, in.rel);
  return vec4f(col, select(1.0, length(in.rel), isReflPass()));
}
`;

// Tileable detail texture: r = grain, g = stones (cellular), b = broad fbm, a = -
export const DETAIL_TEX = NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;
const SIZE: f32 = 512.0;
fn pn(p: vec2f, period: f32) -> f32 {
  // periodic value noise via wrapping the lattice
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21((i % period + period) % period);
  let b = hash21(((i + vec2f(1.0, 0.0)) % period + period) % period);
  let c = hash21(((i + vec2f(0.0, 1.0)) % period + period) % period);
  let d = hash21(((i + vec2f(1.0, 1.0)) % period + period) % period);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn cells(p: vec2f, period: f32) -> f32 {
  let i = floor(p);
  let f = p - i;
  var d1 = 9.0; var d2 = 9.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let o = vec2f(f32(x), f32(y));
      let c = ((i + o) % period + period) % period;
      let r = o + vec2f(hash21(c), hash21(c + 37.0)) - f;
      let d = dot(r, r);
      if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
  }
  return clamp((sqrt(d2) - sqrt(d1)) * 2.5, 0.0, 1.0);
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let uv = (vec2f(id.xy) + 0.5) / SIZE;
  var g = 0.0; var a = 0.5; var f = 32.0;
  for (var o = 0; o < 4; o++) { g += a * pn(uv * f, f); f *= 2.0; a *= 0.5; }
  let st = cells(uv * 24.0, 24.0) * (0.7 + 0.3 * pn(uv * 64.0, 64.0));
  var b = 0.0; a = 0.5; f = 4.0;
  for (var o = 0; o < 4; o++) { b += a * pn(uv * f, f); f *= 2.0; a *= 0.5; }
  textureStore(dst, id.xy, vec4f(g, st, b, 1.0));
}
`;
