// Geometry worker: builds building meshes off the main thread.
import { parseBuildings, buildBuildingMeshes } from './buildings.js';

self.onmessage = (e) => {
  try {
    const list = parseBuildings(e.data.buffer);
    const r = buildBuildingMeshes(list);
    const transfer = [r.lights.buffer];
    for (const m of r.meshes) transfer.push(m.vertices.buffer, m.indices.buffer);
    self.postMessage({ ok: true, ...r }, transfer);
  } catch (err) {
    self.postMessage({ ok: false, error: String(err && err.stack || err) });
  }
};
