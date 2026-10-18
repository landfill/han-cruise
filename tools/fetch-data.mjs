// Downloads the raw geodata the scene is built from into data-raw/ (cached; re-run is cheap).
//   OSM (ODbL) via the Overpass API: river polygon, bridges, buildings, parks, roads.
//   Elevation: AWS Terrain Tiles (terrarium encoding), see THIRD_PARTY_NOTICES.md.
// Usage: node tools/fetch-data.mjs [--force]
import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

// curl is used for HTTP: it behaves consistently behind the sandboxes/firewalls this was built in.
const run = promisify(execFile);
async function curl(args) {
  const { stdout } = await run('curl', ['-sS', '-m', '300', '-A', UA, ...args], { maxBuffer: 1 << 30, encoding: 'buffer' });
  return stdout;
}

const OUT = path.resolve('data-raw');
fs.mkdirSync(OUT, { recursive: true });
const FORCE = process.argv.includes('--force');
const UA = 'han-cruise-dev/0.1 (local research build; contact: repository owner)';
const ENDPOINTS = ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter'];

// Corridor around the route (full building detail) and the wider skyline area.
const CORRIDOR = '37.500,126.905,37.548,126.990';
const WIDE = '37.465,126.860,37.585,127.070';

const QUERIES = {
  river: `[out:json][timeout:180];rel(152336);way(r)(37.470,126.860,37.580,127.060);out geom;`,
  water: `[out:json][timeout:120];(way["natural"="water"](${CORRIDOR});rel["natural"="water"](${CORRIDOR});way["waterway"="riverbank"](${CORRIDOR}););out geom;`,
  bridges: `[out:json][timeout:180];(way["man_made"="bridge"](${CORRIDOR});rel["man_made"="bridge"](${CORRIDOR});way["bridge"]["highway"](${CORRIDOR});way["bridge"]["railway"](${CORRIDOR}););out geom;`,
  buildings_corridor: `[out:json][timeout:240];(way["building"](${CORRIDOR});rel["building"](${CORRIDOR}););out geom;`,
  buildings_wide: `[out:json][timeout:240];(way["building"]["height"](${WIDE});way["building"]["building:levels"](${WIDE});rel["building"]["height"](${WIDE});rel["building"]["building:levels"](${WIDE}););out geom;`,
  landmarks: `[out:json][timeout:120];(rel(8824257);rel(16474080);way["man_made"="tower"](37.548,126.985,37.554,126.992););out geom;`,
  landuse: `[out:json][timeout:180];(way["leisure"="park"](${CORRIDOR});rel["leisure"="park"](${CORRIDOR});way["landuse"~"grass|forest|recreation_ground"](${CORRIDOR});way["natural"~"wood|scrub|sand|beach"](${CORRIDOR});rel["natural"="wood"](${CORRIDOR}););out geom;`,
  roads: `[out:json][timeout:180];(way["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link"](${CORRIDOR});way["railway"~"rail|subway"](${CORRIDOR});way["highway"~"cycleway|footway|path"]["bicycle"](${CORRIDOR}););out geom;`,
  supports: `[out:json][timeout:120];(nwr["bridge:support"](${CORRIDOR});nwr["man_made"="bridge"]["bridge:structure"](${CORRIDOR}););out geom;`,
  // riverside park facilities (points): kiosks, toilets, shelters, cafes, bicycle rental
  facilities: `[out:json][timeout:120];(node["amenity"~"toilets|shelter|cafe|fast_food|restaurant|bicycle_rental|drinking_water"](${CORRIDOR});node["shop"="convenience"](${CORRIDOR}););out;`,
  piers: `[out:json][timeout:120];(way["man_made"="pier"](${CORRIDOR});nwr["amenity"="ferry_terminal"](${CORRIDOR});way["man_made"~"breakwater|embankment|dyke"](${CORRIDOR});way["natural"="tree_row"](${CORRIDOR}););out geom;`,
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function overpass(name, q) {
  const file = path.join(OUT, name + '.json');
  if (!FORCE && fs.existsSync(file)) { console.log('cached', name); return; }
  let wait = 5000;
  for (let attempt = 0; attempt < 10; attempt++) {
    const ep = ENDPOINTS[attempt % ENDPOINTS.length];
    try {
      const text = (await curl(['-H', 'Accept: application/json', '--data-urlencode', 'data=' + q, ep])).toString('utf8');
      if (text.trimStart().startsWith('{')) {
        const j = JSON.parse(text);
        if (j.remark && /runtime error|timed out/i.test(j.remark)) throw new Error(j.remark);
        fs.writeFileSync(file, text);
        console.log('ok', name, j.elements.length, 'elements', (text.length / 1e6).toFixed(1) + 'MB');
        return;
      }
      console.log('retry', name, text.slice(0, 80).replace(/\s+/g, ' '), ep);
    } catch (e) {
      console.log('retry', name, String(e).slice(0, 120));
    }
    await sleep(wait);
    wait = Math.min(wait * 2, 60000);
  }
  throw new Error('failed ' + name);
}

// Terrarium tiles: elevation = (R * 256 + G + B / 256) - 32768 metres.
function lon2tile(lon, z) { return Math.floor(((lon + 180) / 360) * 2 ** z); }
function lat2tile(lat, z) {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z);
}
async function tiles(z, s, w, n, e) {
  const dir = path.join(OUT, 'terrarium', String(z));
  fs.mkdirSync(dir, { recursive: true });
  const x0 = lon2tile(w, z), x1 = lon2tile(e, z), y0 = lat2tile(n, z), y1 = lat2tile(s, z);
  let count = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const f = path.join(dir, `${x}_${y}.png`);
      if (!FORCE && fs.existsSync(f)) continue;
      const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
      fs.writeFileSync(f, await curl(['-f', url]));
      count++;
    }
  }
  console.log(`terrarium z${z}: x ${x0}-${x1} y ${y0}-${y1}, downloaded ${count}`);
}

for (const [name, q] of Object.entries(QUERIES)) {
  await overpass(name, q);
  await sleep(1500);
}
// near: z14 (~9.5 m/px), background mountains: z11 over ~60 km
await tiles(14, 37.465, 126.860, 37.585, 127.070);
await tiles(11, 37.20, 126.60, 37.85, 127.40);
