// Opaque meshes: buildings, bridges, piers, the boat. One vertex format and one shader with
// procedural materials (windows, curtain walls, concrete, painted steel, deck) so every
// structure is lit by the same sun, sky, city light and shadows.
import { LIGHTING_WGSL } from './lighting.js';

// Vertex: pos f32x3 | normal snorm8x4 | colour unorm8x4 | material u8x4 (id, rough, metal, seed) | aux f32x2 (base y, floor height)
export const MESH_STRIDE = 32;
export const MAT = {
  PBR: 0, OFFICE: 1, RESIDENTIAL: 2, HOUSE: 3, CURTAIN: 4, GOLD: 5, ROOF: 6, CONCRETE: 7,
  STEEL: 8, LAMP: 9, DECK: 10, HULL: 11, CABIN_GLASS: 12, PARC1: 13, ASPHALT: 14, RUBBER: 15, OBSTRUCTION: 16,
};

const MESH_IO = /* wgsl */ `
struct Obj { model: mat4x4f, params: vec4f };
@group(1) @binding(0) var<uniform> O: Obj;
@group(1) @binding(1) var detailTex: texture_2d<f32>;

struct VIn {
  @location(0) pos: vec3f,
  @location(1) nrm: vec4f,
  @location(2) col: vec4f,
  @location(3) mat: vec4u,
  @location(4) aux: vec2f,
};
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) world: vec3f,
  @location(2) nrm: vec3f,
  @location(3) col: vec4f,
  @location(4) @interpolate(flat) mat: vec4u,
  @location(5) aux: vec2f,
  @location(6) lpos: vec3f,
};
`;

export const MESH = LIGHTING_WGSL + MESH_IO + /* wgsl */ `
@vertex fn vs(v: VIn) -> VSOut {
  var o: VSOut;
  let w = O.model * vec4f(v.pos, 1.0);
  o.world = w.xyz;
  o.rel = w.xyz - F.camPos.xyz;
  o.nrm = normalize((O.model * vec4f(v.nrm.xyz, 0.0)).xyz);
  o.col = v.col;
  o.mat = v.mat;
  o.aux = v.aux;
  o.lpos = v.pos;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

// Instanced variant (traffic): per-instance position + heading (cw from north) and body colour.
@vertex fn vsInst(v: VIn, @location(5) inst: vec4f, @location(6) tint: vec4f) -> VSOut {
  var o: VSOut;
  let c = cos(inst.w); let s = sin(inst.w);
  // local: x right, z backwards (forward = -z)
  let p = vec3f(c * v.pos.x - s * v.pos.z, v.pos.y, s * v.pos.x + c * v.pos.z);
  let nn = vec3f(c * v.nrm.x - s * v.nrm.z, v.nrm.y, s * v.nrm.x + c * v.nrm.z);
  o.world = inst.xyz + p;
  o.rel = o.world - F.camPos.xyz;
  o.nrm = nn;
  // vertex colour alpha 0 marks body panels that take the instance paint colour
  o.col = vec4f(select(v.col.rgb, tint.rgb, v.col.a < 0.5), 1.0);
  o.mat = v.mat;
  o.aux = v.aux;
  o.lpos = v.pos;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  return o;
}

fn h2(a: f32, b: f32, seed: u32) -> f32 {
  return hashU(u32(i32(a) + 100000) * 7919u + seed, u32(i32(b) + 100000));
}

// detail texture lookup with an explicit LOD from the pixel footprint (usable in any branch)
fn dtex(uv: vec2f, scale: f32, dist: f32) -> vec4f {
  let pixA = 2.0 * length(F.camUp.xyz) * F.screen.w;
  let lod = log2(max(dist * pixA * scale * 512.0, 1.0));
  return textureSampleLevel(detailTex, sampR, uv * scale, lod);
}

struct Facade { alb: vec3f, rough: f32, f0: vec3f, metal: f32, emit: vec3f };

// Window grid on a vertical wall. u along the wall, v up from the base (metres).
// u, v and their screen derivatives are computed by the caller in uniform control flow.
fn facade(u: f32, v: f32, fwu: f32, fwv: f32, floorH: f32, seed: u32, kind: u32, wallCol: vec3f, hasWin: f32, dist: f32) -> Facade {
  var f: Facade;
  let s0 = hashU(seed, 11u);
  let s1 = hashU(seed, 12u);
  var bay = 1.5 + s0 * 1.4;
  var dutyX = 0.55 + s1 * 0.25;
  var dutyY = 0.55;
  if (kind == 2u) { bay = 2.6 + s0 * 1.2; dutyX = 0.62; dutyY = 0.5; }     // apartments: wide balcony windows
  if (kind == 3u) { bay = 2.2 + s0; dutyX = 0.35; dutyY = 0.42; }          // houses / low-rise
  if (kind == 4u || kind == 5u || kind == 13u) { bay = 1.5; dutyX = 0.93; dutyY = 0.86; } // curtain walls
  // some offices have continuous ribbon windows
  if (kind == 1u && hashU(seed, 13u) < 0.3) { dutyX = 0.97; dutyY = 0.45; }
  let cu = u / bay;
  let cv = v / floorH;
  let fw = vec2f(fwu / bay, fwv / floorH);
  // window mask, box-filtered so distant facades average instead of shimmering
  let mx = filteredPulse(cu + 0.5 * (1.0 - dutyX), dutyX, fw.x);
  let my = filteredPulse(cv - 0.25, dutyY, fw.y);
  var win = mx * my * hasWin;
  // ground floor (lobbies / shops) a bit taller and more glazed; no windows at the parapet
  win *= step(0.6, v);
  let cell = vec2f(floor(cu), floor(cv));
  let cellR = h2(cell.x, cell.y, seed);
  // glass: dark interior + reflection (handled by the caller through f0/rough)
  let blinds = select(0.0, cellR, kind == 1u || kind == 2u);
  let interior = mix(vec3f(0.018, 0.02, 0.022), vec3f(0.16, 0.15, 0.13), blinds * 0.6 * (1.0 - F.night.x));
  var glassTint = vec3f(1.0);
  var glassF0 = vec3f(0.045);
  var glassRough = 0.06;
  if (kind == 4u) { glassTint = wallCol; glassF0 = vec3f(0.07) + wallCol * 0.08; glassRough = 0.05; }
  if (kind == 5u) { glassTint = vec3f(1.0, 0.7, 0.3); glassF0 = vec3f(0.95, 0.58, 0.20); glassRough = 0.09; }  // gold-coated reflective glass
  if (kind == 13u) { glassTint = vec3f(0.55, 0.62, 0.65); glassF0 = vec3f(0.08); glassRough = 0.05; }
  // wall / spandrel, with large-scale weathering (rain streaks, patchy repainting)
  let stain = dtex(vec2f(u * 0.04, v * 0.012), 1.0, dist).r;
  let patchv = dtex(vec2f(u * 0.01, v * 0.006), 1.0, dist).b;
  var wall = wallCol * (0.86 + 0.22 * patchv) * mix(1.0, 0.82, smoothstep(0.5, 0.75, stain));
  if (kind == 2u) {
    // horizontal balcony slab lines on apartment blocks
    let slab = filteredPulse(cv + 0.06, 0.12, fw.y);
    wall = mix(wall, wall * 0.75, slab);
  }
  if (kind == 4u) { wall = vec3f(0.12, 0.13, 0.14); }
  if (kind == 5u) { wall = vec3f(0.55, 0.38, 0.16); }
  if (kind == 13u) { wall = vec3f(0.55, 0.06, 0.04); }
  f.alb = mix(wall, interior * glassTint, win);
  // the 63 Building's gold coating also scatters diffusely: it reads gold even against a blue sky
  if (kind == 5u) { f.alb = mix(wall, vec3f(0.40, 0.21, 0.06), win); }
  f.rough = mix(0.8, glassRough, win);
  f.f0 = mix(vec3f(0.035), glassF0, win);
  f.metal = 0.0;

  // lit windows at night. Offices light up in clusters (a lit floor section of several bays),
  // homes per window; the cluster pattern stays readable at a distance, and only beyond that is
  // replaced by its average so distant facades do not sparkle.
  let isHome = kind == 2u || kind == 3u;
  let litFrac = F.night.w * select(0.6, 0.9, isHome);
  let clusterW = select(5.0, 1.0, isHome);
  let ccell = vec2f(floor(cu / clusterW), floor(cv));
  let cR = h2(ccell.x, ccell.y, seed + 7u);
  let wR = select(cellR * 0.5, cellR, isHome);   // some unlit windows inside lit office clusters
  let lit = step(cR, litFrac) * step(wR, 0.85) * step(0.6, v) * hasWin;
  let warm = select(vec3f(1.0, 0.86, 0.66), vec3f(1.0, 0.72, 0.42), isHome);
  let cool = vec3f(0.85, 0.92, 1.0);
  let tone = select(warm, cool, h2(ccell.x + 17.0, ccell.y, seed) > select(0.6, 0.92, isHome));
  let bright = 0.45 + 0.8 * h2(ccell.x, ccell.y + 31.0, seed);
  let one = tone * bright * lit;
  // seen edge-on or from far away, recessed windows are partly hidden by the wall depth
  let avg = mix(warm, cool, 0.3) * litFrac * 0.85 * 0.3;
  let far = smoothstep(0.5, 1.4, max(fw.x * clusterW, fw.y) / clusterW * select(1.0, 1.0, isHome) + max(fw.x, fw.y) * 0.25);
  f.emit = mix(one * win, avg * dutyX * dutyY * hasWin, far) * F.night.y * 0.009;
  return f;
}

fn concrete(world: vec3f, n: vec3f, base: vec3f, dist: f32) -> vec3f {
  let a = abs(n);
  let uv = select(select(world.xy, world.zy, a.x > a.z), world.xz, a.y > 0.7);
  let g = dtex(uv, 0.15, dist).r;
  let b = dtex(uv, 0.02, dist).b;
  // weathering streaks run down vertical faces
  let streak = dtex(vec2f(uv.x * 0.35, world.y * 0.012), 1.0, dist).r;
  return base * (0.82 + 0.25 * g) * (0.85 + 0.3 * b) * mix(1.0, 0.8, smoothstep(0.55, 0.8, streak) * (1.0 - a.y));
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (isReflPass() && in.world.y < -0.2) { discard; }
  var n = normalize(in.nrm);
  // thin two-sided parts (rails, plates) are drawn without culling: face the normal to the viewer
  if (dot(n, in.rel) > 0.0) { n = -n; }
  let id = in.mat.x;
  let rough0 = f32(in.mat.y) / 255.0;
  let metal0 = f32(in.mat.z) / 255.0;
  let seed = in.mat.w + u32(abs(in.aux.x) * 7.0) * 131u;
  let dist = length(in.rel);
  // facade coordinates and derivatives (uniform control flow)
  let tw = normalize(vec2f(-n.z, n.x) + vec2f(1e-5, 0.0));
  let fu = dot(in.world.xz, tw);
  let fv = in.world.y - in.aux.x;
  let fwu = fwidth(fu);
  let fwv = fwidth(fv);
  let deckZ = in.lpos.z / 1.2;
  let fwDeck = fwidth(deckZ);
  var s: Surf;
  s.albedo = in.col.rgb; s.rough = max(rough0, 0.04); s.metal = metal0; s.f0 = vec3f(0.04); s.emit = vec3f(0.0); s.ao = 1.0;
  let isWall = abs(n.y) < 0.3;
  if ((id >= 1u && id <= 5u) || id == 13u) {
    if (isWall) {
      let f = facade(fu, fv, fwu, fwv, max(in.aux.y, 2.5), seed, id, in.col.rgb, step(0.5, in.col.a), dist);
      s.albedo = f.alb; s.rough = f.rough; s.f0 = f.f0; s.metal = f.metal; s.emit = f.emit;
      // contact darkening at the base
      s.ao = mix(0.55, 1.0, smoothstep(0.0, 5.0, in.world.y - in.aux.x));
    } else {
      s.albedo = concrete(in.world, n, vec3f(0.30, 0.30, 0.29), dist); s.rough = 0.85;
    }
  } else if (id == 6u) {
    s.albedo = concrete(in.world, n, in.col.rgb, dist);
  } else if (id == 7u) {
    s.albedo = concrete(in.world, n, in.col.rgb, dist);
    s.rough = 0.88;
  } else if (id == 8u) {
    // painted steel: slight chalking and rust streak variation
    let g = dtex(in.world.xz + in.world.y * 0.66, 0.3, dist).r;
    s.albedo = in.col.rgb * (0.88 + 0.2 * g);
    s.rough = 0.5 + 0.2 * g;
  } else if (id == 9u) {
    s.emit = in.col.rgb * F.night.y * in.col.a * 4.0;
    s.albedo = vec3f(0.6);
  } else if (id == 10u) {
    // deck: anti-slip painted steel deck with faint seams
    let g = dtex(in.lpos.xz, 0.8, dist).r;
    let seam = filteredPulse(deckZ, 0.03, fwDeck);
    s.albedo = in.col.rgb * (0.9 + 0.15 * g) * (1.0 - 0.35 * seam);
    s.rough = 0.65 + 0.15 * g;
  } else if (id == 11u) {
    // hull paint, slightly glossy, dirtier towards the waterline
    let g = dtex(in.lpos.xy, 0.25, dist).r;
    s.albedo = in.col.rgb * (0.95 + 0.07 * g) * mix(0.78, 1.0, smoothstep(0.0, 1.0, in.world.y));
    s.rough = 0.32 + 0.1 * g;
  } else if (id == 12u) {
    s.albedo = vec3f(0.02, 0.025, 0.03); s.f0 = vec3f(0.05); s.rough = 0.04;
    s.emit = vec3f(1.0, 0.85, 0.62) * F.night.y * 0.006;
  } else if (id == 14u) {
    let g = dtex(in.world.xz, 0.5, dist).r;
    s.albedo = vec3f(0.07, 0.07, 0.075) * (0.85 + 0.3 * g); s.rough = 0.8;
  } else if (id == 16u) {
    // aviation obstruction light: slow red blink
    let blink = step(0.5, fract(F.camPos.w * 0.5 + f32(in.mat.w) * 0.013));
    s.emit = vec3f(1.0, 0.05, 0.02) * F.night.y * blink * 3.0;
    s.albedo = vec3f(0.3, 0.02, 0.02);
  }
  // the boat's own deck lamps (params.x): warm fill light on its decks and railings at night
  s.emit += s.albedo * vec3f(1.0, 0.84, 0.62) * F.night.y * O.params.x * (0.4 + 0.6 * max(n.y, 0.0));
  var col = shade(s, n, in.rel, in.world);
  col = applyFog(col, in.rel);
  return vec4f(col, select(1.0, length(in.rel), isReflPass()));
}
`;

// Depth-only pass for shadow cascades (shared by meshes; trees have their own).
export const MESH_SHADOW = /* wgsl */ `
struct SP { vp: mat4x4f, origin: vec4f };
@group(0) @binding(0) var<uniform> SPU: SP;
struct Obj { model: mat4x4f, params: vec4f };
@group(1) @binding(0) var<uniform> O: Obj;
@vertex fn vs(@location(0) pos: vec3f) -> @builtin(position) vec4f {
  let w = O.model * vec4f(pos, 1.0);
  return SPU.vp * vec4f(w.xyz - SPU.origin.xyz, 1.0);
}
`;
