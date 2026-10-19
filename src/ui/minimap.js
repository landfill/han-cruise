// Route minimap for the HUD (wide screens): the river outline drawn from the river SDF, the real
// bridge lines, the voyage path and the boat. The map is turned so this reach of the river lies
// left-to-right (Yeouido below, Yongsan above); the north arrow shows the true north.

const ANGLE = 50 * Math.PI / 180;      // map +x = this compass bearing below east (river axis)
const U0 = -720, U1 = 1300;             // map extent along the river (m)
const V0 = -540, V1 = 580;              // map extent across the river (m, + = north-east bank)
const WIDTH = 340;                      // CSS px
const SVGNS = 'http://www.w3.org/2000/svg';

const BRIDGES = [
  { keys: ['wonhyo'], name: '원효대교' },
  { keys: ['rail_c', 'rail_d', 'rail_a', 'rail_b'], name: '한강철교' },
];
const PLACES = [
  { name: '여의도', u: -60, v: -505 },
  { name: '용산', u: 640, v: 500 },
  { name: '노들섬', u: 1290, v: 255, anchor: 'end' },
];

export function createMinimap(host, { meta, sdf, path }) {
  const c = Math.cos(ANGLE), s = Math.sin(ANGLE);
  const k = WIDTH / (U1 - U0);
  const W = WIDTH, H = Math.round((V1 - V0) * k);
  // world (x east, z south) -> map px
  const toMap = (x, z) => [((x * c + z * s) - U0) * k, (V1 - (x * s - z * c)) * k];
  // map px -> world
  const toWorld = (px, py) => {
    const u = px / k + U0, v = V1 - py / k;
    return [u * c + v * s, u * s - v * c];
  };

  host.style.width = W + 'px';
  host.style.height = H + 'px';

  // --- water: SDF thresholded per device pixel, with an anti-aliased bank line
  const cv = document.createElement('canvas');
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  host.appendChild(cv);
  const sm = meta.sdf;
  const field = new Int16Array(sdf.buffer, sdf.byteOffset, sdf.byteLength / 2);
  const sample = (x, z) => {
    const gx = (x - sm.x0) / sm.cell - 0.5, gz = (z - sm.z0) / sm.cell - 0.5;
    const i = Math.max(0, Math.min(sm.w - 2, Math.floor(gx))), j = Math.max(0, Math.min(sm.h - 2, Math.floor(gz)));
    const fx = Math.min(1, Math.max(0, gx - i)), fz = Math.min(1, Math.max(0, gz - j));
    const o = j * sm.w + i;
    const a = field[o] + (field[o + 1] - field[o]) * fx;
    const b = field[o + sm.w] + (field[o + sm.w + 1] - field[o + sm.w]) * fx;
    return (a + (b - a) * fz) * sm.scale;
  };
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(cv.width, cv.height);
  const mpp = 1 / (k * dpr); // metres per device pixel
  const water = [96, 132, 150, 110], bank = [190, 208, 214, 150];
  for (let py = 0; py < cv.height; py++) {
    for (let px = 0; px < cv.width; px++) {
      const [x, z] = toWorld((px + 0.5) / dpr, (py + 0.5) / dpr);
      const d = sample(x, z) / mpp;
      const wa = Math.min(1, Math.max(0, 0.5 - d));
      const la = Math.max(0, 1 - Math.abs(d) / 0.9);
      const o = (py * cv.width + px) * 4;
      // bank line over water fill
      const aw = (water[3] / 255) * wa, al = (bank[3] / 255) * la;
      const A = al + aw * (1 - al);
      if (A <= 0) continue;
      for (let ch = 0; ch < 3; ch++) img.data[o + ch] = (bank[ch] * al + water[ch] * aw * (1 - al)) / A;
      img.data[o + 3] = A * 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // --- vectors
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  host.appendChild(svg);
  const el = (tag, attrs, parent = svg) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const [a, v] of Object.entries(attrs)) e.setAttribute(a, v);
    parent.appendChild(e);
    return e;
  };
  const text = (str, x, y, cls, anchor = 'middle') => {
    const t = el('text', { x: x.toFixed(1), y: y.toFixed(1), class: cls, 'text-anchor': anchor });
    t.textContent = str;
    return t;
  };

  for (const p of PLACES) {
    const [x, y] = [(p.u - U0) * k, (V1 - p.v) * k];
    text(p.name, x, y, 'mm-place', p.anchor);
  }
  {
    const [x, y] = [(330 - U0) * k, (V1 + 250) * k];
    text('한 강', x, y, 'mm-river');
  }

  // bridges: only the part over the water
  for (const b of BRIDGES) {
    let top = null;
    for (const key of b.keys) {
      const br = meta.bridges.find((q) => q.key === key);
      if (!br) continue;
      const [ax, az] = br.a, [bx, bz] = br.b;
      const L = Math.hypot(bx - ax, bz - az);
      const at = (t) => toMap(ax + (bx - ax) * t / L, az + (bz - az) * t / L);
      const w = br.water || [0, L];
      const p0 = at(w[0]), p1 = at(w[1]);
      el('line', { x1: p0[0].toFixed(1), y1: p0[1].toFixed(1), x2: p1[0].toFixed(1), y2: p1[1].toFixed(1), class: 'mm-bridge' });
      const hi = p0[1] < p1[1] ? p0 : p1;
      if (!top || hi[1] < top[1]) top = hi;
    }
    if (top) text(b.name, Math.min(W - 4, top[0] + 6), Math.max(12, top[1] + 14), 'mm-label', 'start');
  }

  // route: whole path dashed, the part already sailed solid
  let d = '';
  const stepN = Math.max(1, Math.round(8 / path.step));
  for (let i = 0; i < path.x.length; i += stepN) {
    const [x, y] = toMap(path.x[i], path.z[i]);
    d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1);
  }
  {
    const n = path.x.length - 1;
    const [x, y] = toMap(path.x[n], path.z[n]);
    d += 'L' + x.toFixed(1) + ' ' + y.toFixed(1);
  }
  el('path', { d, class: 'mm-route' });
  const done = el('path', { d, class: 'mm-done', pathLength: path.length.toFixed(1), 'stroke-dasharray': `0 ${path.length + 1}` });

  // the pier and the turning point
  const [sx, sy] = toMap(path.x[0], path.z[0]);
  el('circle', { cx: sx.toFixed(1), cy: sy.toFixed(1), r: 4, class: 'mm-stop' });
  text('여의도 선착장', sx + 9, sy + 4, 'mm-label', 'start');
  let far = 0, farU = -Infinity;
  for (let i = 0; i < path.x.length; i++) {
    const u = path.x[i] * c + path.z[i] * s;
    if (u > farU) { farU = u; far = i; }
  }
  const [tx, ty] = toMap(path.x[far], path.z[far]);
  el('circle', { cx: tx.toFixed(1), cy: ty.toFixed(1), r: 3.2, class: 'mm-stop' });

  // north arrow: world north is (0, -1)
  {
    const nx = W - 22, ny = H - 26;
    const dx = -s, dy = -c; // map direction of world north (screen y down)
    const g = el('g', { class: 'mm-north', transform: `translate(${nx} ${ny})` });
    el('line', { x1: (-dx * 9).toFixed(1), y1: (-dy * 9).toFixed(1), x2: (dx * 9).toFixed(1), y2: (dy * 9).toFixed(1) }, g);
    const ang = Math.atan2(dy, dx) * 180 / Math.PI;
    el('path', { d: 'M13 0 L6 -4 L6 4 Z', transform: `rotate(${ang.toFixed(1)})` }, g);
    const t = el('text', { x: (-dx * 13).toFixed(1), y: (-dy * 13 + 4).toFixed(1), 'text-anchor': 'middle', class: 'mm-n' }, g);
    t.textContent = 'N';
  }

  // boat
  const boatG = el('g', { class: 'mm-boat' });
  el('circle', { r: 7, class: 'mm-boat-halo' }, boatG);
  el('path', { d: 'M6 0 L-4 -4 L-2 0 L-4 4 Z' }, boatG);

  return {
    update(boat) {
      done.setAttribute('stroke-dasharray', `${Math.max(0, boat.s).toFixed(1)} ${path.length + 1}`);
      const [x, y] = toMap(boat.x, boat.z);
      // heading: 0 = north, clockwise; world direction (sin h, -cos h)
      const wx = Math.sin(boat.heading), wz = -Math.cos(boat.heading);
      const du = wx * c + wz * s, dv = wx * s - wz * c;
      const ang = Math.atan2(-dv, du) * 180 / Math.PI;
      boatG.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${ang.toFixed(1)})`);
    },
  };
}
