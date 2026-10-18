// Loads scene data with byte-level progress; gzip payloads are inflated with DecompressionStream.
export async function fetchWithProgress(url, onBytes, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export async function gunzip(bytes) {
  // some servers (including Vite's dev server) send .gz files with Content-Encoding: gzip, in
  // which case the browser has already inflated them: check the gzip magic number first
  if (bytes.length < 2 || bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Loads scene.json and every binary it references. progress(fraction 0..1, label)
export async function loadScene(base, progress, signal) {
  const meta = await (await fetch(base + 'scene.json', { signal })).json();
  const files = {
    sdf: meta.sdf.file, landclass: meta.landclass.file, height: meta.height.file,
    far: meta.far.file, buildings: meta.buildings.file, trees: meta.trees.file, lamps: meta.lamps.file,
  };
  // approximate sizes for the progress bar (exact sizes come from the responses)
  let loaded = 0;
  const total = 12.2e6;
  const out = { meta };
  await Promise.all(Object.entries(files).map(async ([k, f]) => {
    const gz = await fetchWithProgress(base + f, (b) => { loaded += b; progress(Math.min(0.99, loaded / total), '지형·건물 데이터 받는 중'); }, signal);
    out[k] = await gunzip(gz);
  }));
  progress(1, '데이터 준비 완료');
  return out;
}
