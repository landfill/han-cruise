// Han River surface. Structure follows web-ocean's ocean shader (c) 2026 Maksymilian Dendura,
// MIT licence: CDLOD mesh, LEAN slope moments, anisotropic Gaussian sun glitter with Smith masking.
// New here: turbid river body colour, surface current, planar reflections of the city with
// depth-aware reprojection, the boat wake simulation (height, turbulent foam, calm trail),
// bridge shadow / overhead occlusion, and the shallow waterline at the revetments.
import { LIGHTING_WGSL } from './lighting.js';
import { CASCADES } from './common.js';
import { KELVIN_WGSL } from './wake.js';

export const WATER = LIGHTING_WGSL + /* wgsl */ `
@group(1) @binding(0) var dispTex: texture_2d_array<f32>;
@group(1) @binding(1) var derivTex: texture_2d_array<f32>;
@group(1) @binding(2) var momTex: texture_2d_array<f32>;
@group(1) @binding(3) var foamTex: texture_2d<f32>;
@group(1) @binding(4) var pathTex: texture_2d<f32>;
@group(1) @binding(5) var reflTex: texture_2d<f32>;
@group(1) @binding(6) var sdfTex: texture_2d<f32>;
@group(1) @binding(7) var sampW: sampler;
@group(1) @binding(8) var<storage, read> passTime: array<f32>;
` + KELVIN_WGSL + /* wgsl */ `
const TEX_N: f32 = 256.0;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) xz: vec2f,
  @location(2) height: f32,
};

fn advected(xz: vec2f) -> vec2f { return xz - F.current.xy * F.current.z; }

@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) inst: vec4f) -> VSOut {
  let n = u32(F.lodP.y);
  let g = vec2f(f32(vi % (n + 1u)), f32(vi / (n + 1u)));
  let spacing = inst.z / f32(n);
  var xz = inst.xy + g * spacing;
  let R = F.lodP.x * exp2(inst.w);
  let d = distance(vec3f(xz.x, 0.0, xz.y), F.camPos.xyz);
  let mk = clamp((d - 0.8 * R) / (0.2 * R), 0.0, 1.0);
  xz -= fract(g * 0.5) * 2.0 * spacing * mk;
  let eff = spacing * (1.0 + mk);
  let ax = advected(xz);
  var disp = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let texel = F.cascL[c] / TEX_N;
    let lod = log2(max(eff / texel, 1.0)) + 0.5;
    if (lod < 7.0) {
      disp += textureSampleLevel(dispTex, sampW, ax * F.cascInvL[c], c, lod).xyz;
    }
  }
  if (distance(xz, F.boat.xy) < 420.0) { disp.y += boatWake(xz, false).h; }
  var o: VSOut;
  let world = vec3f(xz.x + disp.x, disp.y, xz.y + disp.z);
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.xz = xz;
  o.height = disp.y;
  return o;
}

fn fresnelWater(c: f32) -> f32 {
  let x = 1.0 - clamp(c, 0.0, 1.0);
  let x2 = x * x;
  return 0.02 + 0.98 * x2 * x2 * x;
}
fn lambdaSmith(w: vec3f, cov: vec3f) -> f32 {
  let xz2 = max(dot(w.xz, w.xz), 1e-8);
  let il = inverseSqrt(xz2);
  let az = w.xz * il;
  let sig2 = az.x * az.x * cov.x + az.y * az.y * cov.y + 2.0 * az.x * az.y * cov.z;
  let cotT = w.y * il;
  let a = max(cotT, 1e-3) / sqrt(2.0 * max(sig2, 1e-7));
  if (a >= 1.6) { return 0.0; }
  return (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a);
}
fn sdfWater(xz: vec2f) -> f32 {
  let uv = (xz - F.sdfP.xy) / F.sdfP.zw;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return -100.0; }
  return textureSampleLevel(sdfTex, sampC, uv, 0.0).r;
}
// project a camera-relative point into the planar reflection texture
fn reflUV(relp: vec3f) -> vec2f {
  let c = F.reflViewProj * vec4f(relp, 1.0);
  return c.xy / max(c.w, 1e-4) * vec2f(0.5, -0.5) + 0.5;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let xz = in.xz;
  let ax = advected(xz);
  // boat wake (Kelvin waves, turbulent wash, calm trail)
  var wk = WakeOut(0.0, vec2f(0.0), 0.0, 0.0);
  if (distance(xz, F.boat.xy) < 700.0) { wk = boatWake(xz, true); }
  let wslope = wk.slope;
  // the turbulent trail damps the short ripples (the glossy stripe behind a boat)
  let calm = clamp(wk.calm, 0.0, 1.0);
  // cat's paws: patches where gusts roughen the short ripples
  let gustN = textureSample(foamTex, sampR, ax * 0.006 + F.windP.xy * (F.camPos.w * 0.0025)).b;
  let gust = mix(0.45, 1.35, smoothstep(0.32, 0.72, gustN)) * F.windP.w + (1.0 - F.windP.w);

  var s = vec2f(0.0);
  var jxx = 0.0; var jzz = 0.0; var jxz = 0.0;
  var cov = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let uv = ax * F.cascInvL[c];
    var gk = select(1.0, gust, c >= 2u);
    gk *= select(1.0, 1.0 - 0.8 * calm, c >= 1u);
    let dv = textureSample(derivTex, sampW, uv, c) * gk;
    let mv = textureSample(momTex, sampW, uv, c) * (gk * gk);
    let av = textureSample(dispTex, sampW, uv, c) * gk;
    s += dv.xy;
    jxx += dv.z; jzz += dv.w; jxz += av.w;
    cov += vec3f(max(mv.x - dv.x * dv.x, 0.0), max(mv.y - dv.y * dv.y, 0.0), mv.z - dv.x * dv.y);
  }
  let dist = length(in.rel);
  let V = -in.rel / dist;
  let Ld = F.sunDir.xyz;

  let unres = F.waterCol.w * mix(1.0, 0.25, calm) * gust + 0.0016 * smoothstep(40.0, 900.0, dist);
  cov.x += unres;
  cov.y += unres;
  let jx = max(1.0 + jxx, 0.2);
  let jz = max(1.0 + jzz, 0.2);
  let sl = vec2f(s.x / jx, s.y / jz) + wslope;
  let nrm = normalize(vec3f(-sl.x, 1.0, -sl.y));
  let dsx = dpdx(sl);
  let dsy = dpdy(sl);
  let aa = vec3f(dsx.x * dsx.x + dsy.x * dsy.x, dsx.y * dsx.y + dsy.y * dsy.y, dsx.x * dsx.y + dsy.x * dsy.y) * 0.5;
  cov += aa * min(1.0, 0.03 / max(max(aa.x, aa.y), 1e-8));

  let world = vec3f(xz.x, in.height, xz.y);
  let sh = shadowAt(in.rel, vec3f(0.0, 1.0, 0.0));
  let occ = overheadOcclusion(world);
  let sunE = sunLight() * sh * occ;

  // ---- sun glitter ----
  let Hv = normalize(V + Ld);
  let hs = -Hv.xz / max(Hv.y, 1e-3);
  let dsl = hs - sl;
  let det = max(cov.x * cov.y - cov.z * cov.z, 1e-10);
  let e = (dsl.x * dsl.x * cov.y + dsl.y * dsl.y * cov.x - 2.0 * dsl.x * dsl.y * cov.z) / det;
  let pdf = exp(-0.5 * e) / (2.0 * PI * sqrt(det));
  let hy2 = Hv.y * Hv.y;
  let Dm = pdf / max(hy2 * hy2, 1e-6);
  let G = 1.0 / (1.0 + lambdaSmith(V, cov) + lambdaSmith(Ld, cov));
  let Fh = fresnelWater(dot(V, Hv));
  var spec = sunE * (Fh * Dm * G / (4.0 * max(V.y, 0.02))) * step(0.0, Ld.y);
  // moon glade at night
  let Hm = normalize(V + F.moonDir.xyz);
  let hm = -Hm.xz / max(Hm.y, 1e-3) - sl;
  let em = (hm.x * hm.x * cov.y + hm.y * hm.y * cov.x - 2.0 * hm.x * hm.y * cov.z) / det;
  let Dmoon = exp(-0.5 * em) / (2.0 * PI * sqrt(det)) / max(Hm.y * Hm.y * Hm.y * Hm.y, 1e-6);
  spec += moonLight() * occ * fresnelWater(dot(V, Hm)) * Dmoon * G / (4.0 * max(V.y, 0.02));

  // ---- reflection: planar reflection of the city/sky, reprojected through the wave normal ----
  let sig = sqrt(max(cov.x + cov.y, 1e-6) * 0.5);
  let vh = normalize(vec3f(V.x, 0.0, V.z) + vec3f(1e-5, 0.0, 0.0));
  let sigV = sqrt(max(vh.x * vh.x * cov.x + vh.z * vh.z * cov.y + 2.0 * vh.x * vh.z * cov.z, 1e-6));
  let aV = max(V.y, 1e-3) / max(sqrt(1.0 - V.y * V.y), 1e-3) / (1.4142 * sigV);
  let tilt = 0.32 * sigV * (1.0 - smoothstep(0.0, 2.5, aV));
  let nV = normalize(nrm + vh * tilt);
  var R = reflect(-V, nV);
  R.y = abs(R.y);
  R = normalize(R);
  let relP = vec3f(in.rel.x, -F.camPos.y, in.rel.z);  // surface point at mean level
  let uv0 = reflUV(relP);
  let hit0 = textureSampleLevel(reflTex, sampC, uv0, 0.0);
  // distance from the surface to whatever the reflected ray hits (alpha = path length in the reflection pass)
  let camToP = length(relP);
  let D = clamp(hit0.a - camToP, 2.0, 6000.0);
  let uv = reflUV(relP + R * D);
  let pixW = 2.0 * length(F.camUp.xyz) * F.screen.w * 2.0;    // reflection texel angle (half res)
  // only the unresolved (sub-pixel) roughness blurs the reflection: resolved wave slopes are
  // already in the per-pixel reflection direction, which is what stretches lights into streaks
  let sigU = sqrt(max(unres + 0.5 * (aa.x + aa.y), 1e-7));
  let blurW = sigU * 2.0 * D / max(pixW * (D + camToP), 1e-5);
  let lod = clamp(log2(max(blurW * 0.35, 1.0)), 0.0, min(F.misc.z - 1.0, 2.5));
  var refl = textureSampleLevel(reflTex, sampC, clamp(uv, vec2f(0.001), vec2f(0.999)), lod).rgb;
  // outside the reflection frustum fall back to the env cube
  let inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
  let envLod = clamp(log2(max(2.0 * sig * ENV_TEXELS_PER_RAD, 1.0)), 0.0, ENV_MAXMIP);
  let envR = textureSampleLevel(envCube, sampC, R, envLod).rgb * occ;
  let edge = smoothstep(0.0, 0.03, min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y)));
  refl = mix(envR, refl, inside * edge);
  let nF = normalize(mix(nrm, vec3f(0.0, 1.0, 0.0), 0.45));
  let Fr = fresnelWater(max(dot(nF, V), 0.0));

  // ---- water body: turbid river (silt + algae), lit by sky and sun ----
  let skyUp = textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb;
  let skyHemi = skyUp * 0.45 + 0.1375 * (
      textureSampleLevel(envCube, sampC, vec3f(0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(-0.94, 0.34, 0.0), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, 0.94), ENV_MAXMIP - 1.0).rgb +
      textureSampleLevel(envCube, sampC, vec3f(0.0, 0.34, -0.94), ENV_MAXMIP - 1.0).rgb);
  // the view angle changes how much of the turbid column we see: steeper looks lighter/greener
  let steep = smoothstep(0.05, 0.6, V.y);
  var bodyCol = F.waterCol.rgb * mix(0.85, 1.15, steep);
  var body = bodyCol * (skyHemi * 1.5 * occ + sunE * max(Ld.y, 0.0) * 0.75 + cityAmbient(vec3f(0.0, 1.0, 0.0), world) * 2.0);
  // light scattered forward through ripple crests when looking towards a low sun
  let toSun = max(dot(Ld, -V), 0.0);
  body += F.sssCol.rgb * sunE * F.sssCol.w * pow(toSun, 5.0) * clamp(in.height * 8.0 + 0.3, 0.0, 1.0) * exp(-dist / 300.0);

  // shallow water at the revetment: the submerged riprap shows through for a few decimetres
  let dB = sdfWater(xz);
  let depth = max(-(max(-6.0, dB * 0.45 - 0.25)), 0.0);
  let shallow = exp(-depth / 0.35) * smoothstep(-8.0, -0.2, dB);
  body = mix(body, vec3f(0.035, 0.034, 0.028) * (skyHemi * 2.0 * occ + sunE * max(Ld.y, 0.0) * 0.6), shallow * 0.6);

  // ---- wake foam (turbulent white water aft, bow wave crests) ----
  let fo = clamp(wk.foam * F.foamP.x, 0.0, 1.0);
  var foamMask = 0.0;
  var foamLit = vec3f(0.0);
  let pat = textureSample(foamTex, sampR, ax * 0.33 + vec2f(0.013, 0.0) * F.camPos.w).r * 0.75 + textureSample(foamTex, sampR, ax * 0.9).g * 0.25;
  let fw = fwidth(pat) + 0.04;
  if (fo > 0.003) {
    // ageing wash breaks into lacy patches; only the fresh core is dense
    foamMask = smoothstep(1.0 - fo - fw - 0.12, 1.0 - fo + fw + 0.12, pat) * smoothstep(0.0, 0.35, fo) * mix(0.55, 0.9, smoothstep(0.5, 0.9, fo));
    // river foam is beige-grey, not ocean white
    let foamAlb = vec3f(0.55, 0.57, 0.52);
    foamLit = foamAlb * (sunE * max(dot(nrm, Ld), 0.0) * 0.32 + skyHemi * 1.4 * occ + cityAmbient(vec3f(0.0, 1.0, 0.0), world) * 3.0);
    // aerated water just under the foam is lighter and greener
    body += vec3f(0.03, 0.05, 0.04) * skyHemi * fo * 1.5;
  }

  var col = body * (1.0 - Fr) + refl * Fr + spec * (1.0 - foamMask);
  col = mix(col, foamLit, foamMask * 0.9);
  if (F.post.w > 0.5) {
    if (F.post.w < 1.5) { return vec4f(refl, 1.0); }
    if (F.post.w < 2.5) { return vec4f(body * (1.0 - Fr), 1.0); }
    if (F.post.w < 3.5) { return vec4f(vec3f(wk.h * 2.0 + 0.5, wk.foam, wk.calm), 1.0) * 0.3; }
    return vec4f(vec3f(Fr), 1.0) * 0.5;
  }
  col = applyFog(col, in.rel);
  return vec4f(col, 1.0);
}
`;

// Water height under a few points (camera / bow / stern), read back for the boat's heave.
export const HEIGHT_PROBE = /* wgsl */ `
struct PU { p: array<vec4f, 4>, cascL: vec4f, cascInvL: vec4f, current: vec4f };
@group(0) @binding(0) var<uniform> U: PU;
@group(0) @binding(1) var dispTex: texture_2d_array<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> outH: array<f32, 4>;
fn dispAt(xz: vec2f) -> vec3f {
  var d = vec3f(0.0);
  let a = xz - U.current.xy * U.current.z;
  for (var c = 0u; c < ${CASCADES}u; c++) {
    d += textureSampleLevel(dispTex, samp, a * U.cascInvL[c], c, 0.0).xyz;
  }
  return d;
}
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_index) i: u32) {
  let goal = U.p[i].xy;
  var x0 = goal;
  for (var k = 0; k < 3; k++) { x0 = goal - dispAt(x0).xz; }
  outH[i] = dispAt(x0).y;
}
`;
