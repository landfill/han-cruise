// Car and bus templates for the traffic (one mesh each, instanced with a paint colour).
// Local frame: -z forward, +x right, y up from the road surface. Body panels use colour alpha 0
// so the shader substitutes the instance's paint.
import { MeshBuilder } from './meshbuilder.js';
import { MAT } from '../wgsl/mesh.js';

const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];

export function buildCarTemplate() {
  const mb = new MeshBuilder(512);
  const body = (c, hu, hv, hw) => { mb.material(MAT.HULL, [1, 1, 1], 0.3, 0.2, 0, 0); mb.box(c, X, Y, Z, hu, hv, hw); };
  body([0, 0.62, 0], 0.88, 0.32, 2.25);                 // lower body
  mb.material(MAT.CABIN_GLASS, [0.05, 0.06, 0.07], 0.05);
  mb.box([0, 1.18, 0.25], X, Y, Z, 0.8, 0.26, 1.15);     // glasshouse
  body([0, 1.46, 0.3], 0.78, 0.03, 0.95);               // roof
  mb.material(MAT.RUBBER, [0.03, 0.03, 0.03], 0.9);
  for (const z of [-1.45, 1.4]) for (const x of [-0.82, 0.82]) mb.box([x, 0.33, z], X, Y, Z, 0.12, 0.33, 0.33);
  return mb.result();
}

export function buildBusTemplate() {
  const mb = new MeshBuilder(512);
  mb.material(MAT.HULL, [1, 1, 1], 0.35, 0.1, 0, 0);
  mb.box([0, 1.75, 0], X, Y, Z, 1.25, 1.3, 5.5);
  mb.material(MAT.CABIN_GLASS, [0.05, 0.06, 0.07], 0.05);
  for (const x of [-1.26, 1.26]) mb.box([x, 2.05, 0.2], X, Y, Z, 0.02, 0.55, 4.8);
  mb.box([0, 2.0, -5.51], X, Y, Z, 1.1, 0.75, 0.02);
  mb.material(MAT.RUBBER, [0.03, 0.03, 0.03], 0.9);
  for (const z of [-3.6, 3.4]) for (const x of [-1.1, 1.1]) mb.box([x, 0.45, z], X, Y, Z, 0.15, 0.45, 0.5);
  return mb.result();
}

// Typical Korean traffic: mostly white, black, grey and silver cars
export const CAR_COLOURS = [
  [0.85, 0.85, 0.84], [0.85, 0.85, 0.84], [0.04, 0.04, 0.045], [0.04, 0.04, 0.045], [0.55, 0.56, 0.57],
  [0.32, 0.33, 0.35], [0.7, 0.71, 0.72], [0.12, 0.17, 0.3], [0.45, 0.08, 0.06], [0.83, 0.82, 0.78],
];
// Seoul city buses: blue trunk lines, green branch lines
export const BUS_COLOURS = [[0.08, 0.32, 0.65], [0.22, 0.55, 0.22]];
