// Sky background, trees, light sprites (lamps, traffic, obstruction lights) and auto exposure.
import { LIGHTING_WGSL } from './lighting.js';
import { SKY_CLOUD_FUNCS } from './sky.js';

// Full-resolution sky + clouds (adapted from web-ocean's sky pass, MIT).
export const SKY_BG = LIGHTING_WGSL + SKY_CLOUD_FUNCS + /* wgsl */ `
struct VO { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dir = normalize(F.camFwd.xyz + in.ndc.x * F.camRight.xyz + in.ndc.y * F.camUp.xyz);
  let pixAng = 2.0 * length(F.camUp.xyz) * F.screen.w;
  var d2 = dir;
  d2.y = max(d2.y, 0.0005);
  d2 = normalize(d2);
  let hl = hitLayer(d2, F.cloudP.x);
  let el = lowEdge(hl.q, hl.t * pixAng / hl.mu, 7);
  let hh = hitLayer(d2, 4.5);
  let eh = highEdge(hh.q, hh.t * pixAng / hh.mu, 6);
  let aa = vec2f(fwidth(el), fwidth(eh));
  var sunT = 1.0;
  let cl = cloudLayers(d2, pixAng, aa, &sunT);
  var L = skyAtmo(d2) * (1.0 - cl.a) + cl.rgb + sunDisk(dir) * sunT + moonDisk(dir) * sunT;
  // below the horizon (only visible past the edge of the terrain): hazy ground
  if (dir.y < 0.0) { L = mix(L, skyAtmo(normalize(vec3f(dir.x, 0.002, dir.z))) * 0.7, smoothstep(0.0, -0.05, dir.y)); }
  return vec4f(L, select(1.0, 20000.0, isReflPass()));
}
`;

// Composites the half-resolution sky into the main pass (only where nothing else was drawn).
export const SKY_COMPOSITE = /* wgsl */ `
@group(0) @binding(0) var skyTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VO { @builtin(position) pos: vec4f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VO;
  o.pos = vec4f(p, 0.0, 1.0);
  return o;
}
@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(skyTex)) * 2.0;
  return vec4f(textureSampleLevel(skyTex, samp, in.pos.xy / dims, 0.0).rgb, 1.0);
}
`;

// Low-poly trees, instanced. Crown normals are blended towards the sphere normal so the
// canopy reads as a soft mass; backlit foliage transmits light.
export const TREES = LIGHTING_WGSL + /* wgsl */ `
@group(1) @binding(0) var detailTex: texture_2d<f32>;
struct VIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec3f,
  @location(2) part: f32,          // 0 trunk, 1 crown
  @location(3) inst: vec4f,        // x, z, ground y, scale
  @location(4) rnd: f32,
};
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) world: vec3f,
  @location(2) nrm: vec3f,
  @location(3) @interpolate(flat) info: vec2f,
  @location(4) lp: vec3f,
};
@vertex fn vs(v: VIn) -> VSOut {
  var o: VSOut;
  let a = v.rnd * 6.2831;
  let c = cos(a); let s = sin(a);
  let sc = v.inst.w * vec3f(1.0 + 0.25 * fract(v.rnd * 7.0), 1.0 + 0.35 * fract(v.rnd * 13.0), 1.0);
  var p = v.pos * sc;
  p = vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
  // gentle sway of the crowns
  let sway = v.part * sin(F.camPos.w * 1.3 + v.inst.x * 0.05 + v.inst.y * 0.07) * 0.04 * p.y / 8.0;
  let world = vec3f(v.inst.x + p.x + sway, v.inst.z + p.y, v.inst.y + p.z);
  o.world = world;
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  let nn = vec3f(c * v.nrm.x - s * v.nrm.z, v.nrm.y, s * v.nrm.x + c * v.nrm.z);
  o.nrm = nn;
  o.info = vec2f(v.part, v.rnd);
  o.lp = v.pos;
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (isReflPass() && in.world.y < -0.2) { discard; }
  var n = normalize(in.nrm);
  let g = textureSample(detailTex, sampR, in.world.xz * 0.9 + in.world.y * 0.37).r;
  var s: Surf;
  s.rough = 0.9; s.metal = 0.0; s.f0 = vec3f(0.03); s.emit = vec3f(0.0); s.ao = 1.0;
  if (in.info.x < 0.5) {
    s.albedo = vec3f(0.10, 0.08, 0.06);
  } else {
    // mid-October: mostly green with some trees turning (zelkova / ginkgo)
    let r = in.info.y;
    var leaf = mix(vec3f(0.07, 0.11, 0.035), vec3f(0.11, 0.13, 0.04), fract(r * 3.7));
    if (fract(r * 11.3) > 0.86) { leaf = mix(vec3f(0.30, 0.22, 0.04), vec3f(0.32, 0.14, 0.04), fract(r * 5.1)); }
    s.albedo = leaf * (0.7 + 0.6 * g);
    // interior of the crown is darker (self-occlusion)
    s.ao = clamp(0.45 + 0.6 * (in.lp.y - 2.0) / 7.0, 0.35, 1.0) * (0.75 + 0.35 * g);
    n = normalize(n + vec3f(g - 0.5) * 0.8);
  }
  var col = shade(s, n, in.rel, in.world);
  if (in.info.x > 0.5) {
    // light transmitted through the leaves when the sun is behind the tree
    let v = normalize(-in.rel);
    let back = pow(max(dot(-v, F.sunDir.xyz), 0.0), 4.0);
    col += s.albedo * vec3f(1.2, 1.4, 0.6) * sunLight() * shadowAt(in.rel, n) * back * 0.35;
  }
  col = applyFog(col, in.rel);
  return vec4f(col, select(1.0, length(in.rel), isReflPass()));
}
`;

export const TREES_SHADOW = /* wgsl */ `
struct SP { vp: mat4x4f, origin: vec4f };
@group(0) @binding(0) var<uniform> SPU: SP;
@vertex fn vs(@location(0) pos: vec3f, @location(3) inst: vec4f, @location(4) rnd: f32) -> @builtin(position) vec4f {
  let a = rnd * 6.2831;
  let c = cos(a); let s = sin(a);
  let sc = inst.w * vec3f(1.0 + 0.25 * fract(rnd * 7.0), 1.0 + 0.35 * fract(rnd * 13.0), 1.0);
  var p = pos * sc;
  p = vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
  let w = vec3f(inst.x + p.x, inst.z + p.y, inst.y + p.z);
  return SPU.vp * vec4f(w - SPU.origin.xyz, 1.0);
}
`;

// Point lights drawn as camera-facing glows. Sub-pixel lights keep their energy (the quad is
// clamped to ~1.5 px and dimmed accordingly), so distant rows of lamps neither vanish nor sparkle.
export const SPRITES = LIGHTING_WGSL + /* wgsl */ `
struct VIn {
  @location(0) p: vec4f,     // world xyz, size (m)
  @location(1) c: vec4f,     // rgb radiance scale, directional flag/dir yaw (w)
};
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec3f,
  @location(2) rel: vec3f,
};
@vertex fn vs(@builtin(vertex_index) vi: u32, v: VIn) -> VSOut {
  var o: VSOut;
  let q = vec2f(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
  let rel = v.p.xyz - F.camPos.xyz;
  let dist = length(rel);
  // pixel footprint of the light at this distance
  let pixWorld = dist * 2.0 * length(F.camUp.xyz) * F.screen.w;
  let minR = pixWorld * 1.6;
  let r = max(v.p.w, minR);
  let energy = (v.p.w * v.p.w) / (r * r);
  var dirK = 1.0;
  if (v.c.w > -9.0) {
    // directional emitter (car head/tail lights): yaw in radians, cw from north
    let fwd = vec3f(sin(v.c.w), 0.0, -cos(v.c.w));
    dirK = smoothstep(-0.1, 0.6, dot(fwd, -rel / max(dist, 1e-3)));
  }
  let right = normalize(F.camRight.xyz);
  let up = normalize(F.camUp.xyz);
  let world = v.p.xyz + (right * q.x + up * q.y) * r * 2.5;
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  // pull the glow slightly towards the camera so it is not swallowed by its own fixture
  o.pos.z = o.pos.z * 1.002;
  o.uv = q * 2.5;
  o.col = v.c.rgb * energy * dirK * F.night.y;
  return o;
}
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (isReflPass() && in.rel.y + F.camPos.y < -0.2) { discard; }
  let d2 = dot(in.uv, in.uv);
  // core + a small halo (scatter in the lens/atmosphere); bloom adds the rest
  let core = exp(-d2 * 2.2);
  let halo = exp(-d2 * 0.45) * 0.06;
  let dist = length(in.rel);
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist * 0.8);
  let c = in.col * (core + halo) * T;
  return vec4f(c, 0.0);
}
`;

// Exposure adaptation from the bloom chain's smallest level (≈ scene average).
export const EXPOSURE = /* wgsl */ `
struct EU { p: vec4f };   // x = dt, y = key (target mid grey), z = min exposure, w = max exposure
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> U: EU;
@group(0) @binding(2) var<storage, read_write> ex: array<f32, 4>;
@compute @workgroup_size(1)
fn main() {
  let dims = vec2i(textureDimensions(src));
  var acc = 0.0;
  var wsum = 0.0;
  for (var j = 0; j < dims.y; j++) {
    for (var i = 0; i < dims.x; i++) {
      let c = textureLoad(src, vec2i(i, j), 0).rgb;
      let l = max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-6);
      // centre-weighted, log average (bright sky does not dominate)
      let uv = (vec2f(f32(i), f32(j)) + 0.5) / vec2f(dims) - 0.5;
      let w = 1.0 - 0.6 * dot(uv, uv) * 2.0;
      acc += log2(l) * w;
      wsum += w;
    }
  }
  let avg = exp2(acc / max(wsum, 1e-4));
  let tgt = clamp(U.p.y / avg, U.p.z, U.p.w);
  let cur = ex[0];
  var e = tgt;
  if (cur > 0.0 && U.p.x > 0.0) {
    // adapt in log space, ~1.5 s time constant
    let k = 1.0 - exp(-U.p.x / 1.5);
    e = exp2(mix(log2(cur), log2(tgt), k));
  }
  ex[0] = e;
  ex[1] = avg;
}
`;
