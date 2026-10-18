// Verifies the demonstration route against the processed OSM data:
//   - every point of the path stays in water with clearance for the hull (half beam + margin)
//   - each bridge crossing passes between piers (distance to the nearest pier)
//   - prints bridge order along the route and landmark coordinates for routeData.js
import fs from 'node:fs';
import zlib from 'node:zlib';
import { Path } from '../src/sim/route.js';
import { ROUTE_POINTS } from '../src/sim/routeData.js';
import { bridgeFrame } from '../src/scene/bridges.js';
import { buildBoat } from '../src/scene/boat.js';

// highest point of the boat (mast) above the waterline
const boatGeo = buildBoat();
const AIR_DRAFT = boatGeo.mesh.max[1];

const meta = JSON.parse(fs.readFileSync('public/data/scene.json', 'utf8'));
const s = meta.sdf;
const raw = zlib.gunzipSync(fs.readFileSync('public/data/' + s.file));
const sdf = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);
const at = (x, z) => {
  const fx = (x - s.x0) / s.cell - 0.5, fz = (z - s.z0) / s.cell - 0.5;
  const i = Math.floor(fx), j = Math.floor(fz), tx = fx - i, tz = fz - j;
  const g = (a, b) => sdf[b * s.w + a] * s.scale;
  return (g(i, j) * (1 - tx) + g(i + 1, j) * tx) * (1 - tz) + (g(i, j + 1) * (1 - tx) + g(i + 1, j + 1) * tx) * tz;
};
const path = new Path(ROUTE_POINTS, 2);
console.log(`route length ${path.length.toFixed(0)} m`);
let worst = Infinity, worstS = 0;
for (let i = 0; i < path.x.length; i++) {
  const d = -at(path.x[i], path.z[i]);
  // the berth itself is next to the pier: skip the first/last 40 m
  const sArc = i * path.step;
  if (sArc < 40 || sArc > path.length - 40) continue;
  if (d < worst) { worst = d; worstS = sArc; }
}
console.log(`min distance to bank (excluding berth): ${worst.toFixed(1)} m at s=${worstS.toFixed(0)} (hull half-beam 5.5 m)`);

let ok = worst > 20;
for (const br of meta.bridges) {
  const [ax, az] = br.a, [bx, bz] = br.b;
  const L = Math.hypot(bx - ax, bz - az), ux = (bx - ax) / L, uz = (bz - az) / L;
  for (let i = 1; i < path.x.length; i++) {
    // signed side of the bridge axis
    const sd = (x, z) => (x - ax) * uz - (z - az) * ux;
    const s0 = sd(path.x[i - 1], path.z[i - 1]), s1 = sd(path.x[i], path.z[i]);
    if (Math.sign(s0) === Math.sign(s1)) continue;
    const t = s0 / (s0 - s1);
    const x = path.x[i - 1] + (path.x[i] - path.x[i - 1]) * t, z = path.z[i - 1] + (path.z[i] - path.z[i - 1]) * t;
    const along = (x - ax) * ux + (z - az) * uz;
    if (along < 0 || along > L) continue;
    const nearest = Math.min(...br.piers.map((p) => Math.abs(p - along)));
    const heading = (Math.atan2(path.x[i] - path.x[i - 1], -(path.z[i] - path.z[i - 1])) * 180 / Math.PI + 360) % 360;
    const axisAz = (Math.atan2(ux, -uz) * 180 / Math.PI + 360) % 180;
    let cross = Math.abs(((heading - axisAz) % 180 + 180) % 180);
    const prof = br.profile[Math.round(along / br.profStep)];
    // underside of the structure at the crossing (girder depth for box girders, floor beams for trusses)
    let depth = 1.4;
    if (br.type === 'boxgirder' || br.type === 'hangang') {
      const p = br.piers;
      let k = 0;
      while (k < p.length - 2 && p[k + 1] < along) k++;
      const t = (along - p[k]) / (p[k + 1] - p[k]);
      depth = br.depthMid + (br.depthPier - br.depthMid) * Math.abs(2 * t - 1) ** 2.6;
    }
    const under = bridgeFrame(br).deck(along) - depth;
    const clear = under - AIR_DRAFT;
    console.log(`s=${(i * path.step).toFixed(0).padStart(5)}  ${br.name.padEnd(10)} station ${along.toFixed(0)} m, nearest pier ${nearest.toFixed(1)} m, crossing ${cross.toFixed(0)} deg, deck ${prof} m, underside ${under.toFixed(1)} m, clearance over mast ${clear.toFixed(1)} m`);
    if (nearest < 18 || clear < 1.0) ok = false;
  }
}
const proj = meta.projection;
const xz = (lat, lon) => [((lon - proj.lon0) * proj.kx).toFixed(0), (-(lat - proj.lat0) * proj.kz).toFixed(0)];
console.log('landmarks:', {
  '63': xz(37.51985, 126.94025), parc1: xz(37.52668, 126.92719), ifc: xz(37.52524, 126.92577),
  nodeul: xz(37.5175, 126.9585), nSeoul: xz(37.551169, 126.988227),
});
console.log(ok ? 'ROUTE OK' : 'ROUTE CHECK FAILED');
process.exit(ok ? 0 : 1);
