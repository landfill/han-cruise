// A do-nothing WebGPU for measuring the app's main-thread cost without a GPU (tools/jank.mjs).
// Injected before the page scripts. Every WebGPU object is a stub: calls return at once,
// promises resolve at once, mapped ranges are zero-filled. It counts the API calls per frame
// (window.__gpuCalls) so the encoding work can be compared between builds.
// ?gpulog: also hashes every call (method, arguments, uploaded bytes) into window.__gpuHash, so
// two builds can be checked for sending the GPU exactly the same work.
// What it leaves out: the browser's own cost of each call (serialising it to the GPU process)
// and all GPU work. Main-thread JavaScript, GC, layout and the compositor still run for real.
(() => {
  const calls = { n: 0 };
  window.__gpuCalls = calls;
  const LOG = /[?&]gpulog/.test(location.search);
  let hash = 0x811c9dc5, nextId = 1;
  const ids = new WeakMap();
  const mix = (x) => { hash = Math.imul(hash ^ x, 16777619) >>> 0; };
  const mixStr = (t) => { for (let i = 0; i < t.length; i++) mix(t.charCodeAt(i)); };
  function mixVal(v, depth = 0) {
    if (v === null || v === undefined) { mix(0); return; }
    if (typeof v === 'number') { mixStr(String(v)); return; }
    if (typeof v === 'string' || typeof v === 'boolean') { mixStr(String(v)); return; }
    if (typeof v === 'function' || (typeof v === 'object' && ids.has(v))) { mixStr('#' + (ids.get(v) || 0)); return; }
    if (ArrayBuffer.isView(v)) { const b = new Uint8Array(v.buffer, v.byteOffset, v.byteLength); for (let i = 0; i < b.length; i++) mix(b[i]); return; }
    if (v instanceof ArrayBuffer) { mixVal(new Uint8Array(v), depth); return; }
    if (depth > 4) return;
    if (Array.isArray(v)) { for (const x of v) mixVal(x, depth + 1); return; }
    if (typeof v === 'object') for (const k of Object.keys(v)) { mixStr(k); mixVal(v[k], depth + 1); }
  }
  // writeBuffer(buffer, offset, data, dataOffset, size): hash only the bytes written
  function logCall(name, args) {
    if (!LOG) return;
    mixStr(name);
    if (name === 'writeBuffer' && ArrayBuffer.isView(args[2])) {
      const d = args[2], el = d.BYTES_PER_ELEMENT || 1, off = args[3] || 0, n = args[4] ?? (d.length - off);
      mixVal(args[0]); mixVal(args[1]); mixVal(new Uint8Array(d.buffer, d.byteOffset + off * el, n * el));
    } else for (const a of args) mixVal(a);
    window.__gpuHash = hash;
  }
  const consts = (o) => new Proxy(o, { get: (t, k) => (k in t ? t[k] : 0) });
  // the usage/stage/map-mode flag objects exist only when WebGPU is enabled
  const flags = (names) => { const o = {}; names.forEach((n, i) => { o[n] = 1 << i; }); return o; };
  if (!window.GPUBufferUsage) window.GPUBufferUsage = flags(['MAP_READ', 'MAP_WRITE', 'COPY_SRC', 'COPY_DST', 'INDEX', 'VERTEX', 'UNIFORM', 'STORAGE', 'INDIRECT', 'QUERY_RESOLVE']);
  if (!window.GPUTextureUsage) window.GPUTextureUsage = flags(['COPY_SRC', 'COPY_DST', 'TEXTURE_BINDING', 'STORAGE_BINDING', 'RENDER_ATTACHMENT']);
  if (!window.GPUShaderStage) window.GPUShaderStage = flags(['VERTEX', 'FRAGMENT', 'COMPUTE']);
  if (!window.GPUMapMode) window.GPUMapMode = flags(['READ', 'WRITE']);
  void consts;

  // generic stub: any property is a callable stub, calls return a new stub
  function stub(extra = {}) {
    const target = function () {};
    const p = new Proxy(target, {
      get(t, k) {
        if (k in extra) return extra[k];
        if (k === 'then' || typeof k === 'symbol') return undefined;
        return (...args) => { calls.n++; logCall(k, args); return makeFor(k, args); };
      },
    });
    ids.set(p, nextId++);
    return p;
  }
  function texture(desc) {
    const s = Array.isArray(desc.size) ? desc.size : [desc.size.width, desc.size.height || 1, desc.size.depthOrArrayLayers || 1];
    return stub({ width: s[0], height: s[1] || 1, depthOrArrayLayers: s[2] || 1, mipLevelCount: desc.mipLevelCount || 1, format: desc.format });
  }
  function buffer(desc) {
    let mapped = null;
    return stub({
      size: desc.size,
      mapAsync: () => { calls.n++; return Promise.resolve(); },
      getMappedRange: (off = 0, size) => { calls.n++; mapped = new ArrayBuffer(size ?? desc.size - off); return mapped; },
      unmap: () => { calls.n++; mapped = null; },
      destroy: () => { calls.n++; },
    });
  }
  function makeFor(name, args) {
    switch (name) {
      case 'createTexture': return texture(args[0]);
      case 'createBuffer': return buffer(args[0]);
      case 'createShaderModule': return stub({ getCompilationInfo: () => Promise.resolve({ messages: [] }) });
      case 'createRenderPipelineAsync': case 'createComputePipelineAsync': return Promise.resolve(stub());
      case 'onSubmittedWorkDone': return Promise.resolve();
      default: return stub();
    }
  }
  const features = new Set(['float32-filterable']);
  const limits = new Proxy({}, { get: () => 1 << 30 });
  const queue = stub({ onSubmittedWorkDone: () => Promise.resolve() });
  const device = stub({ features, limits, queue, lost: new Promise(() => {}), createTexture: (d) => { calls.n++; logCall('createTexture', [d]); return texture(d); }, createBuffer: (d) => { calls.n++; logCall('createBuffer', [d]); return buffer(d); }, onuncapturederror: null });
  const adapter = { features, limits, info: { vendor: 'stub', architecture: 'stub' }, requestDevice: () => Promise.resolve(device) };
  Object.defineProperty(navigator, 'gpu', { value: { requestAdapter: () => Promise.resolve(adapter), getPreferredCanvasFormat: () => 'bgra8unorm' }, configurable: true });

  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, ...rest) {
    if (type !== 'webgpu') return getContext.call(this, type, ...rest);
    const canvas = this;
    // keep the canvas changing every frame (as the real swap chain would), so the
    // compositor and anything layered over it (backdrop-filter) do real work
    // compositor and anything layered over it (backdrop-filter) do real work. ?fakefill=full
    // repaints the whole canvas (costs main-thread raster time the real app does not have);
    // by default only a corner changes.
    const ctx2d = getContext.call(canvas, '2d');
    const full = /[?&]fakefill=full/.test(location.search);
    let k = 0;
    return {
      configure() {}, unconfigure() {},
      getCurrentTexture() {
        calls.n++;
        k = (k + 1) & 255;
        if (ctx2d) { ctx2d.fillStyle = `rgb(${k},${(k * 3) & 255},${255 - k})`; ctx2d.fillRect(0, 0, full ? canvas.width : 16, full ? canvas.height : 16); }
        return texture({ size: [canvas.width, canvas.height] });
      },
    };
  };
})();
