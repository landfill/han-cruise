// Distant landmarks that OSM footprints do not capture well: N Seoul Tower on Namsan
// (236.7 m tower on a ~262 m summit) and Lotte World Tower (554.5 m) far to the east.
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

export function buildLandmarks(lm) {
  const mb = new MeshBuilder(4096);
  const lights = [];
  {
    const [x, y, z] = lm.nSeoulTower.pos;
    // base pavilion
    mb.material(MAT.CONCRETE, [0.7, 0.7, 0.68], 0.8);
    mb.lathe(x, y - 2, z, [[16, 0], [16, 9], [14, 11], [6.6, 12]], 20);
    // concrete shaft
    mb.material(MAT.CONCRETE, [0.82, 0.82, 0.8], 0.75);
    mb.lathe(x, y + 10, z, [[6.5, 0], [6.0, 40], [5.2, 115], [5.0, 122]], 16);
    // observation levels
    mb.material(MAT.CURTAIN, [0.3, 0.35, 0.4], 0.1);
    mb.lathe(x, y + 130, z, [[5.0, 0], [12.5, 3], [14.0, 7], [14.0, 15], [12.5, 19], [5.0, 21]], 24);
    // antenna mast
    mb.material(MAT.STEEL, [0.85, 0.85, 0.85], 0.5, 0.3);
    mb.lathe(x, y + 151, z, [[4.6, 0], [4.2, 14], [2.3, 20], [1.6, 50], [0.9, 85.7]], 10);
    lights.push(x, y + 236.5, z, 0.9, 1.0, 0.1, 0.05, -10);
    lights.push(x, y + 145, z, 3.0, 0.7, 0.8, 1.0, -10);
  }
  {
    const [x, y, z] = lm.lotteWorldTower.pos;
    mb.material(MAT.CURTAIN, [0.62, 0.66, 0.68], 0.1, 0, 77);
    mb.setAux(y, 4.0);
    const prof = [];
    for (let k = 0; k <= 16; k++) {
      const t = k / 16;
      prof.push([34 * (1 - 0.72 * t ** 1.7), t * 554.5 - 0.5]);
    }
    prof.push([0.5, 555]);
    mb.lathe(x, y, z, prof, 12, [1, 1]);
    mb.setAux(0, 0);
    lights.push(x, y + 556, z, 2.0, 1.0, 0.1, 0.05, -10);
  }
  return { mesh: mb.result(), lights: new Float32Array(lights) };
}
