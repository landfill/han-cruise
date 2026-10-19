// Procedural bridge structures from the axes, spans and deck profiles computed by
// tools/build-scene.mjs. Types:
//   boxgirder     haunched prestressed-concrete box girder (Wonhyo: FCM/Dywidag, V-shaped piers)
//   truss_warren  simple through Warren trusses (Hangang Railway Bridge lines A and B)
//   truss_cont    continuous through trusses, deeper over the piers (lines C and D)
//   hangang       tied arches on the Nodeul-Noryangjin section, girders on the north section
// Returns mesh data, light sprites (street lamps) and deck lanes for traffic.
import { MeshBuilder, cross, norm } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

const CONCRETE = [0.56, 0.55, 0.52];
const CONCRETE_DARK = [0.46, 0.45, 0.43];
// Hangang Railway Bridge trusses are painted a light green (checked against photographs)
const RAIL_STEEL = [0.30, 0.56, 0.42];
const ARCH_STEEL = [0.78, 0.79, 0.8];

export function bridgeFrame(br) {
  const [ax, az] = br.a, [bx, bz] = br.b;
  const L = Math.hypot(bx - ax, bz - az);
  const u = [(bx - ax) / L, 0, (bz - az) / L];
  const n = [-u[2], 0, u[0]]; // right of the axis
  const deck = (s) => {
    const f = Math.max(0, Math.min(br.profile.length - 1.001, s / br.profStep));
    const i = Math.floor(f), t = f - i;
    return br.profile[i] * (1 - t) + br.profile[i + 1] * t;
  };
  const P = (s, off, y) => [ax + u[0] * s + n[0] * off, y, az + u[2] * s + n[2] * off];
  return { L, u, n, deck, P };
}

// girder depth at station s: haunched (deepest at the piers) across the river spans
function girderDepth(br, s) {
  const p = br.piers;
  const [w0, w1] = br.water;
  if (s < w0 - 1 || s > w1 + 1 || !br.depthPier) return br.depthMid || 2.2;
  let k = 0;
  while (k < p.length - 2 && p[k + 1] < s) k++;
  const span = p[k + 1] - p[k];
  if (span < 45) return br.depthMid;
  const t = (s - p[k]) / span;
  // haunch: deep over the piers, shallow over most of the span (FCM cantilever profile)
  const e = Math.abs(2 * t - 1);
  return br.depthMid + (br.depthPier - br.depthMid) * e ** 2.6;
}

function sweep(mb, fr, s0, s1, ds, sectionFn) {
  // sectionFn(s) -> array of [offset, y] points (closed loop is not assumed; strip between consecutive)
  const stations = [];
  for (let s = s0; s < s1; s += ds) stations.push(s);
  stations.push(s1);
  const secs = stations.map((s) => sectionFn(s).map(([o, y]) => fr.P(s, o, y)));
  const m = secs[0].length;
  for (let j = 0; j < m - 1; j++) {
    for (let k = 0; k < stations.length - 1; k++) {
      const a = secs[k][j], b = secs[k][j + 1], c = secs[k + 1][j + 1], d = secs[k + 1][j];
      mb.quad(a, d, c, b);
    }
  }
}

function lampsOnDeck(br, fr, out, mb, spacing = 32) {
  const half = br.deckW / 2 - 0.4;
  for (let s = 15; s < fr.L - 15; s += spacing) {
    for (const side of [-1, 1]) {
      const y = fr.deck(s);
      const base = fr.P(s, side * half, y);
      const top = fr.P(s, side * half, y + 10);
      const head = fr.P(s, side * (half - 2.2), y + 10.2);
      mb.material(MAT.STEEL, [0.45, 0.46, 0.47], 0.5, 0.6);
      mb.tube(base, top, 0.12, 6);
      mb.beam(top, head, 0.14, 0.14);
      mb.material(MAT.LAMP, [1.0, 0.85, 0.65], 0.5, 0, 0, 0.6);
      mb.box([head[0], head[1] - 0.12, head[2]], fr.u, [0, 1, 0], fr.n, 0.25, 0.08, 0.35);
      out.push(head[0], head[1] - 0.25, head[2], 0.35, 1.0, 0.82, 0.6, -10);
    }
  }
}

function parapets(mb, fr, br) {
  const half = br.deckW / 2;
  mb.material(MAT.CONCRETE, [0.62, 0.61, 0.58], 0.85);
  sweep(mb, fr, 0, fr.L, 10, (s) => {
    const y = fr.deck(s);
    return [[-half, y], [-half, y + 1.05], [-half + 0.4, y + 1.05], [-half + 0.45, y]];
  });
  sweep(mb, fr, 0, fr.L, 10, (s) => {
    const y = fr.deck(s);
    return [[half - 0.45, y], [half - 0.4, y + 1.05], [half, y + 1.05], [half, y]];
  });
}

function deckSurface(mb, fr, br) {
  const half = br.deckW / 2;
  mb.material(MAT.ASPHALT, [0.08, 0.08, 0.085], 0.8);
  sweep(mb, fr, 0, fr.L, 10, (s) => { const y = fr.deck(s); return [[-half, y], [half, y]]; });
}

function boxGirder(mb, fr, br, lights) {
  const W = br.deckW, half = W / 2;
  deckSurface(mb, fr, br);
  parapets(mb, fr, br);
  mb.material(MAT.CONCRETE, CONCRETE, 0.85);
  // cross-section: cantilever slab + trapezoidal box, swept along the axis
  sweep(mb, fr, 0, fr.L, 5, (s) => {
    const y = fr.deck(s);
    const d = girderDepth(br, s);
    const tb = W * 0.27, bb = W * 0.21;
    return [
      [half, y], [half, y - 0.25], [tb + 0.6, y - 0.45], [tb, y - 0.6],
      [bb, y - d], [-bb, y - d], [-tb, y - 0.6], [-tb - 0.6, y - 0.45], [-half, y - 0.25], [-half, y],
    ];
  });
  piersFor(mb, fr, br);
  lampsOnDeck(br, fr, lights, mb);
}

function piersFor(mb, fr, br) {
  const [w0, w1] = br.water;
  for (const s of br.piers) {
    const y = fr.deck(s);
    const d = girderDepth(br, s);
    const top = y - d;
    const inWater = s >= w0 - 2 && s <= w1 + 2;
    const footY = inWater ? -1.5 : Math.min(top - 1, 3.0);
    if (top - footY < 1) continue;
    const c = fr.P(s, 0, 0);
    if (br.pier === 'v' && inWater) {
      // V-shaped pier in elevation (Wonhyo): two legs spreading along the axis
      mb.material(MAT.CONCRETE, CONCRETE, 0.85);
      // caisson cap at the waterline (elliptical, cutwater shaped)
      // round caisson (~13 m) just above the water, a short pedestal, then two thick legs
      // spreading along the bridge axis to the haunched girder (as in photographs of the bridge)
      const c = fr.P(s, 0, 0);
      mb.material(MAT.CONCRETE, [0.58, 0.56, 0.52], 0.9);
      mb.lathe(c[0], footY, c[2], [[6.5, 0], [6.5, 1.5 - footY], [5.8, 1.7 - footY]], 20);
      mb.material(MAT.CONCRETE, CONCRETE, 0.85);
      mb.box(fr.P(s, 0, 3.0), fr.u, [0, 1, 0], fr.n, 2.6, 1.4, br.deckW * 0.22);
      const legW = br.deckW * 0.4;
      for (const dir of [-1, 1]) {
        const a = fr.P(s + dir * 1.2, 0, 3.6), b = fr.P(s + dir * 4.6, 0, top + 0.2);
        mb.beam(a, b, 2.4, legW, fr.n);
      }
      continue;
    }
    mb.material(MAT.CONCRETE, inWater ? CONCRETE : CONCRETE_DARK, 0.85);
    if (inWater) capsule(mb, fr, s, 0, footY, 1.4, 4.0, br.deckW * 0.36);
    const cols = br.deckW > 16 ? [-br.deckW * 0.22, br.deckW * 0.22] : [0];
    for (const off of cols) {
      const a = fr.P(s, off, footY + (inWater ? 1.4 : 0)), b = fr.P(s, off, top);
      mb.beam(a, b, 2.0, 2.0, fr.u);
    }
    if (cols.length > 1) {
      const a = fr.P(s, -br.deckW * 0.3, top - 0.9), b = fr.P(s, br.deckW * 0.3, top - 0.9);
      mb.beam(a, b, 1.8, 1.8);
    }
    void c;
  }
}

// rounded pier footing (pointed upstream/downstream like a cutwater): length along the flow
function capsule(mb, fr, s, off, y0, h, along, across) {
  // the river flows across the bridge axis, so the long side of the footing is along n
  const pts = [];
  const N = 14;
  for (let k = 0; k < N; k++) {
    const t = (k / N) * Math.PI * 2;
    const ca = Math.cos(t), sa = Math.sin(t);
    // superellipse with pointed ends along n
    const rn = across * 0.5 * Math.sign(ca) * Math.abs(ca) ** 0.8;
    const rs = along * 0.5 * Math.sign(sa) * Math.abs(sa) ** 1.2 * 0.45;
    pts.push([rs, rn]);
  }
  const P = (q, y) => fr.P(s + q[0], off + q[1], y);
  for (let k = 0; k < N; k++) {
    const a = pts[k], b = pts[(k + 1) % N];
    mb.quad(P(a, y0), P(b, y0), P(b, y0 + h), P(a, y0 + h));
  }
  const ctr = fr.P(s, off, y0 + h);
  const ci = mb.vertex(ctr[0], ctr[1], ctr[2], 0, 1, 0);
  for (let k = 0; k < N; k++) {
    const a = P(pts[k], y0 + h), b = P(pts[(k + 1) % N], y0 + h);
    mb.tri(ci, mb.vertex(...a, 0, 1, 0), mb.vertex(...b, 0, 1, 0));
  }
}

function truss(mb, fr, br, continuous) {
  const half = br.deckW / 2;
  const H = br.trussH;
  // track bed: deck slab + ballast + rails
  mb.material(MAT.STEEL, [0.2, 0.19, 0.18], 0.7, 0.3);
  sweep(mb, fr, 0, fr.L, 10, (s) => { const y = fr.deck(s); return [[-half, y - 0.4], [-half, y], [half, y], [half, y - 0.4]]; });
  mb.material(MAT.PBR, [0.22, 0.2, 0.18], 0.95);
  sweep(mb, fr, 0, fr.L, 10, (s) => { const y = fr.deck(s); return [[-1.6, y + 0.01], [-1.4, y + 0.3], [1.4, y + 0.3], [1.6, y + 0.01]]; });
  mb.material(MAT.STEEL, [0.45, 0.42, 0.4], 0.35, 0.9);
  for (const o of [-0.72, 0.72]) sweep(mb, fr, 0, fr.L, 10, (s) => { const y = fr.deck(s) + 0.3; return [[o - 0.04, y], [o - 0.04, y + 0.17], [o + 0.04, y + 0.17], [o + 0.04, y]]; });

  const piers = br.piers;
  const [w0, w1] = br.water;
  mb.material(MAT.STEEL, RAIL_STEEL, 0.55, 0.15);
  for (let k = 0; k < piers.length - 1; k++) {
    const sA = piers[k], sB = piers[k + 1];
    const span = sB - sA;
    const overWater = sA >= w0 - 40 && sB <= w1 + 40;
    if (!overWater || span < 40) {
      // approach: plate girders
      sweep(mb, fr, sA, sB, Math.max(5, span / 4), (s) => { const y = fr.deck(s) - 0.4; return [[-half, y], [-half, y - 1.8], [half, y - 1.8], [half, y]]; });
      continue;
    }
    const panels = Math.max(6, Math.round(span / 7.75));
    // continuous trusses (3-span units) are deeper over the interior supports
    const unitPos = k % 3;
    const depthAt = (t) => {
      if (!continuous) return H;
      const atA = unitPos === 1 || unitPos === 2 ? 1 : 0;
      const atB = unitPos === 0 || unitPos === 1 ? 1 : 0;
      return H + 2.5 * (atA * (1 - t) ** 3 + atB * t ** 3);
    };
    for (const side of [-1, 1]) {
      const o = side * half;
      const bot = [], top = [];
      for (let i = 0; i <= panels; i++) {
        const t = i / panels, s = sA + span * t;
        const y = fr.deck(s);
        // end panels slope down to the bearings (classic through-truss portals)
        const endK = Math.min(i, panels - i) === 0 ? 0 : 1;
        bot.push(fr.P(s, o, y - 0.2));
        top.push(endK ? fr.P(s, o, y + depthAt(t)) : null);
      }
      // top chord (from first to last interior node), end posts inclined
      for (let i = 1; i < panels - 1; i++) mb.beam(top[i], top[i + 1], 0.75, 0.75, fr.n);
      mb.beam(bot[0], top[1], 0.7, 0.75, fr.n);
      mb.beam(bot[panels], top[panels - 1], 0.7, 0.75, fr.n);
      // bottom chord
      for (let i = 0; i < panels; i++) mb.beam(bot[i], bot[i + 1], 0.75, 0.9, fr.n);
      // web members
      for (let i = 1; i < panels; i++) {
        if (continuous || i % 2 === 0) mb.beam(bot[i], top[i], 0.38, 0.5, fr.n); // verticals
        if (continuous) {
          // Pratt: diagonals slope down towards mid-span
          if (i < panels / 2) mb.beam(top[i], bot[i + 1], 0.4, 0.5, fr.n);
          else if (i + 1 < panels) mb.beam(top[i + 1], bot[i], 0.4, 0.5, fr.n);
          if (i === Math.floor(panels / 2) && panels % 2 === 1) mb.beam(top[i], bot[i + 1], 0.4, 0.5, fr.n);
        } else {
          // Warren: alternating diagonals
          const j = i % 2 === 1 ? i - 1 : i + 1;
          if (j >= 0 && j <= panels) mb.beam(top[i], bot[j], 0.42, 0.52, fr.n);
        }
      }
      // collect top nodes for bracing
      if (side === -1) br._topL = top; else br._topR = top;
    }
    // top lateral bracing & sway frames
    for (let i = 1; i < panels; i++) {
      const a = br._topL[i], b = br._topR[i];
      if (!a || !b) continue;
      mb.beam(a, b, 0.35, 0.45);
      if (i + 1 < panels && br._topL[i + 1]) {
        mb.beam(a, br._topR[i + 1], 0.22, 0.22);
      }
    }
    // floor beams
    for (let i = 0; i <= panels; i++) {
      const s = sA + span * i / panels;
      mb.beam(fr.P(s, -half, fr.deck(s) - 0.6), fr.P(s, half, fr.deck(s) - 0.6), 0.4, 0.8);
    }
    mb.material(MAT.STEEL, RAIL_STEEL, 0.55, 0.15);
  }
  // round concrete piers with a cap under each support (as photographed)
  for (const s of piers) {
    const y = fr.deck(s) - 1.0;
    const inWater = s >= w0 - 2 && s <= w1 + 2;
    const y0 = inWater ? -1.5 : Math.min(y - 1, 3);
    if (y - y0 < 1) continue;
    const c = fr.P(s, 0, 0);
    mb.material(MAT.CONCRETE, inWater ? [0.55, 0.54, 0.5] : CONCRETE_DARK, 0.9);
    mb.lathe(c[0], y0, c[2], [[2.6, 0], [2.6, y - 0.8 - y0], [3.2, y - 0.6 - y0], [3.2, y - y0]], 16);
    if (inWater) mb.lathe(c[0], -1.5, c[2], [[3.4, 0], [3.4, 1.2], [2.6, 1.6]], 16);
  }
}

function archSpans(mb, fr, br, lights, spans, rise, planesOff, led = null) {
  mb.material(MAT.STEEL, ARCH_STEEL, 0.45, 0.2);
  for (const [sA, sB] of spans) {
    const span = sB - sA;
    const segs = 16;
    const nodes = { [-1]: [], [1]: [] };
    for (const side of [-1, 1]) {
      const o = side * planesOff;
      for (let i = 0; i <= segs; i++) {
        const t = i / segs, s = sA + span * t;
        const y = fr.deck(s) + 0.5 + rise * Math.sin(Math.PI * t);
        nodes[side].push(fr.P(s, o, y));
      }
      for (let i = 0; i < segs; i++) mb.beam(nodes[side][i], nodes[side][i + 1], 0.9, 1.1, fr.n);
      // hangers
      for (let i = 1; i < segs; i++) {
        const s = sA + span * i / segs;
        mb.tube(nodes[side][i], fr.P(s, o, fr.deck(s) + 0.5), 0.06, 4);
      }
    }
    // wind bracing between the ribs near the crown
    for (let i = 4; i <= segs - 4; i += 2) mb.beam(nodes[-1][i], nodes[1][i], 0.4, 0.5);
    if (led) {
      // LED line along the top of each rib (the arches are outlined in light at night)
      mb.material(MAT.LAMP, led, 0.5, 0, 0, 0.35);
      for (const side of [-1, 1]) for (let i = 0; i < segs; i++) {
        const a = nodes[side][i], b = nodes[side][i + 1];
        mb.beam([a[0], a[1] + 0.58, a[2]], [b[0], b[1] + 0.58, b[2]], 0.18, 0.06, fr.n);
      }
      mb.material(MAT.STEEL, ARCH_STEEL, 0.45, 0.2);
    }
    // floodlit arches at night: a few soft lights along the ribs
    for (const side of [-1, 1]) for (let i = 2; i < segs; i += led ? 2 : 3) {
      const p = nodes[side][i];
      lights.push(p[0], p[1] + 0.8, p[2], 0.25, 0.85, 0.9, 1.0, -10);
    }
  }
}

export function buildBridges(bridges) {
  const mb = new MeshBuilder(1 << 16);
  const lights = [];
  const lanes = [];  // for traffic: {pts:[[x,y,z]...], dir}
  const occluders = []; // deck footprints: {a, b, width, profile, profStep, under}
  for (const br of bridges) {
    const fr = bridgeFrame(br);
    if (br.type === 'boxgirder') {
      boxGirder(mb, fr, br, lights);
      if (br.archSpan || br.arches) {
        // main channel arch(es) over the central spans
        const p = br.piers, mid = (br.water[0] + br.water[1]) / 2;
        const spans = [];
        for (let k = 0; k < p.length - 1; k++) if (p[k] >= br.water[0] - 1 && p[k + 1] <= br.water[1] + 1) spans.push([p[k], p[k + 1]]);
        spans.sort((x, y) => Math.abs((x[0] + x[1]) / 2 - mid) - Math.abs((y[0] + y[1]) / 2 - mid));
        let chosen = spans.slice(0, br.arches || 1);
        if (br.archSpan) { const c = spans[0]; const cm = (c[0] + c[1]) / 2; chosen = [[cm - br.archSpan / 2, cm + br.archSpan / 2]]; }
        archSpans(mb, fr, br, lights, chosen, br.archSpan ? 26 : 16, br.deckW / 2 - 1);
      }
    } else if (br.type === 'truss_warren' || br.type === 'truss_cont') {
      truss(mb, fr, br, br.type === 'truss_cont');
    } else if (br.type === 'hangang') {
      // girder deck (shallow steel box), then arches on the southern section
      br.depthMid = 2.2; br.depthPier = 2.6;
      boxGirder(mb, fr, br, lights);
      if (br.key === 'hangang0') {
        const p = br.piers.filter((s) => s >= br.water[0] - 1 && s <= br.water[1] + 1);
        const spans = [];
        for (let k = 0; k < p.length - 1; k++) spans.push([p[k], p[k + 1]]);
        archSpans(mb, fr, br, lights, spans, 11, br.deckW / 2 - 0.8, [0.85, 0.92, 1.0]);
      }
    }
    const roadBridge = !br.type.startsWith('truss');
    if (roadBridge) {
      const nl = Math.max(2, Math.floor(br.deckW / 3.5) & ~1);
      for (let i = 0; i < nl; i++) {
        const off = (i - (nl - 1) / 2) * 3.3;
        const pts = [];
        for (let s = 0; s <= fr.L; s += 20) pts.push(fr.P(s, off, fr.deck(s) + 0.05));
        const dir = off > 0 ? 1 : -1; // keep right: lanes on the right of the axis travel along +u
        lanes.push({ pts: dir > 0 ? pts : pts.reverse(), bridge: br.key });
      }
    }
    occluders.push({ a: br.a, b: br.b, width: br.deckW + (br.type.startsWith('truss') ? 2 : 0), profile: br.profile, profStep: br.profStep, depth: br.depthMid || 1.5, key: br.key });
  }
  return { mesh: mb.result(), lights: new Float32Array(lights), lanes, occluders };
}

export { cross, norm };

// Elevated roads along the banks (viaducts, interchange ramps): deck slab on a shallow girder,
// parapets, single or twin columns every ~30 m, street lamps. pts: [x, deckY, z, groundY].
export function buildViaducts(viaducts) {
  const mb = new MeshBuilder(1 << 15);
  const lights = [];
  const lanes = [];
  for (const v of viaducts) {
    const P = v.pts;
    if (P.length < 2) continue;
    const half = v.width / 2;
    // per-vertex miter normals (right of the direction of travel)
    const nrm = P.map((_, i) => {
      const a = P[Math.max(0, i - 1)], b = P[Math.min(P.length - 1, i + 1)];
      const dx = b[0] - a[0], dz = b[2] - a[2], l = Math.hypot(dx, dz) || 1;
      return [-dz / l, dx / l];
    });
    const at = (i, off, dy) => [P[i][0] + nrm[i][0] * off, P[i][1] + dy, P[i][2] + nrm[i][1] * off];
    const strip = (prof) => {
      for (let i = 0; i < P.length - 1; i++) {
        for (let k = 0; k < prof.length - 1; k++) {
          const [o0, y0] = prof[k], [o1, y1] = prof[k + 1];
          mb.quad(at(i, o0, y0), at(i + 1, o0, y0), at(i + 1, o1, y1), at(i, o1, y1));
        }
      }
    };
    mb.material(MAT.ASPHALT, [0.08, 0.08, 0.085], 0.8);
    strip([[-half, 0], [half, 0]]);
    mb.material(MAT.CONCRETE, [0.55, 0.54, 0.51], 0.85);
    strip([[half, 0], [half, -0.35], [half * 0.55, -1.4], [-half * 0.55, -1.4], [-half, -0.35], [-half, 0]]);
    strip([[-half, 0], [-half, 1.0], [-half + 0.35, 1.0], [-half + 0.35, 0]]);
    strip([[half - 0.35, 0], [half - 0.35, 1.0], [half, 1.0], [half, 0]]);
    // columns and lamps along the length
    let acc = 0;
    for (let i = 0; i < P.length - 1; i++) {
      const L = Math.hypot(P[i + 1][0] - P[i][0], P[i + 1][2] - P[i][2]);
      for (let s = (30 - acc) % 30; s < L; s += 30) {
        const t = s / L;
        const x = P[i][0] + (P[i + 1][0] - P[i][0]) * t, z = P[i][2] + (P[i + 1][2] - P[i][2]) * t;
        const y = P[i][1] + (P[i + 1][1] - P[i][1]) * t, g = P[i][3] + (P[i + 1][3] - P[i][3]) * t;
        const n = nrm[i];
        if (y - 1.4 - g > 1.5) {
          mb.material(MAT.CONCRETE, [0.5, 0.49, 0.46], 0.9);
          const cols = v.width > 14 ? [-half * 0.4, half * 0.4] : [0];
          for (const off of cols) mb.tube([x + n[0] * off, g - 0.5, z + n[1] * off], [x + n[0] * off, y - 1.3, z + n[1] * off], 0.9, 8);
        }
        // lamp on the right parapet
        const lx = x + n[0] * (half - 0.2), lz = z + n[1] * (half - 0.2);
        mb.material(MAT.STEEL, [0.45, 0.46, 0.47], 0.5, 0.6);
        mb.tube([lx, y + 1, lz], [lx, y + 10, lz], 0.1, 5);
        lights.push(lx - n[0] * 1.8, y + 9.8, lz - n[1] * 1.8, 0.33, 1.0, 0.8, 0.55, -10);
      }
      acc = (acc + L) % 30;
    }
    // traffic: lanes 3.4 m apart, keep right
    const nl = Math.max(1, Math.floor(v.width / 3.6));
    for (let k = 0; k < nl; k++) {
      const off = (k - (nl - 1) / 2) * 3.4;
      const pts = P.map((_, i) => at(i, off, 0.05));
      if (v.oneway) lanes.push({ pts, speed: 18 });
      else lanes.push({ pts: off > 0 ? pts : pts.slice().reverse(), speed: 16 });
    }
  }
  return { mesh: mb.result(), lights: new Float32Array(lights), lanes };
}
