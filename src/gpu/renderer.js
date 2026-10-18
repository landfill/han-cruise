// WebGPU renderer for the Han River scene.
// Pass order per frame:
//   compute: FFT waves, sky-view LUT, sun visibility, env cube face (+mips), height probe
//   (the boat wake is analytic, evaluated in the water shader from a route field built at load)
//   shadow:  3 sun cascades (buildings, bridges, boat, trees)
//   reflect: mirrored camera at half resolution (land, buildings, bridges, boat, trees, sky, lights)
//   main:    4x MSAA HDR (land, structures, boat, trees, water, sky, lights)
//   post:    bloom, adaptive exposure, AgX tone mapping
// Per-frame code reuses preallocated typed arrays; GPU timing via timestamp queries.
import { FRAME, CASCADES, SKY_W, SKY_H, ENV_SIZE, ENV_MIPS } from '../wgsl/common.js';
import { simInit, simEvolve, fftRows, fftCols, SIM_MIPS, FOAM_TEX, RGBA8_MIPS, RGBA16_MIPS } from '../wgsl/sim.js';
import { TRANS_LUT, MS_LUT, SKYVIEW_LUT, ENV_CUBE, CUBE_MIPS, SUN_VIS } from '../wgsl/sky.js';
import { WATER, HEIGHT_PROBE } from '../wgsl/water.js';
import { TERRAIN, DETAIL_TEX } from '../wgsl/terrain.js';
import { MESH, MESH_SHADOW, MESH_STRIDE } from '../wgsl/mesh.js';
import { SKY_BG, SKY_COMPOSITE, TREES, TREES_SHADOW, SPRITES, EXPOSURE } from '../wgsl/extras.js';
import { BLOOM_DOWN, BLOOM_UP, COMPOSITE } from '../wgsl/post.js';
import { PATH_FIELD } from '../wgsl/wake.js';
import { CDLOD, gridIndices } from './cdlod.js';
import { mat4, mul, perspective, viewFromBasis, translation, toHalf } from './math.js';
import { TREE_STRIDE } from '../scene/trees.js';

const GU = () => GPUBufferUsage;
const TU = () => GPUTextureUsage;
const SS = () => GPUShaderStage;
const HDR = 'rgba16float';
const MSAA = 4;
const BLOOM_LEVELS = 7;
const CASC_L = [41.0, 9.7, 2.3, 0.53];   // FFT patch sizes (m): river chop, ripples, capillaries
const SHADOW_FAR = [70, 600, 3600];
const LOOP = 600;

export const QUALITY = {
  low: { fftN: 128, shadow: 1024, reflScale: 0.33, scale: 0.7, treeDist: 900, label: '낮음' },
  medium: { fftN: 256, shadow: 2048, reflScale: 0.5, scale: 0.85, treeDist: 1500, label: '보통' },
  high: { fftN: 256, shadow: 2048, reflScale: 0.5, scale: 1.0, treeDist: 2500, label: '높음' },
};

export async function createRenderer({ device, context, format, canvas, data, geom, quality = 'high', log = console.log }) {
  const U = GU(), T = TU(), S = SS();
  const Q = { ...QUALITY[quality] };
  const N = Q.fftN;
  const meta = data.meta;
  const errors = [];
  device.onuncapturederror = (e) => { errors.push(e.error.message); log('GPU error: ' + e.error.message); };

  // ------------------------------------------------------------------------------------------
  // helpers
  // ------------------------------------------------------------------------------------------
  async function shader(code, label) {
    const m = device.createShaderModule({ code, label });
    const info = await m.getCompilationInfo();
    let bad = false;
    for (const msg of info.messages) {
      if (msg.type === 'error') bad = true;
      if (msg.type !== 'info') {
        const line = code.split('\n')[msg.lineNum - 1];
        log(`[${label}] ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}\n    ${line}`);
      }
    }
    if (bad) throw new Error('셰이더 컴파일 오류: ' + label);
    return m;
  }
  const computePipe = async (code, label) => device.createComputePipelineAsync({ label, layout: 'auto', compute: { module: await shader(code, label), entryPoint: 'main' } });
  const buf = (size, usage, label) => device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, label });
  const bgAuto = (pipe, entries, label) => device.createBindGroup({ label, layout: pipe.getBindGroupLayout(0), entries: entries.map((r, i) => ({ binding: i, resource: r })) });
  const lvl = (t, m, dim = '2d-array') => t.createView({ dimension: dim, baseMipLevel: m, mipLevelCount: 1 });
  const fullArr = (t) => t.createView({ dimension: '2d-array' });
  const upload = (arr, usage, label) => {
    const b = device.createBuffer({ size: Math.ceil(arr.byteLength / 4) * 4, usage: usage | U.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    b.unmap();
    return b;
  };

  // ------------------------------------------------------------------------------------------
  // samplers & shared textures
  // ------------------------------------------------------------------------------------------
  const sampClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const sampRepeat = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 4 });
  // Wave textures: trilinear. Anisotropic filtering of the 12 cascade lookups cost ~6 ms per frame
  // on the reference GPU; the LEAN slope moments already turn unresolved detail into roughness.
  const sampWater = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
  const sampShadow = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
  const linSamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

  function halfTexture(src, w, h, scale, label) {
    const tex = device.createTexture({ label, size: [w, h], format: 'r16float', usage: T.TEXTURE_BINDING | T.COPY_DST });
    const rowBytes = Math.ceil((w * 2) / 256) * 256;
    const staging = new Uint16Array((rowBytes / 2) * h);
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) staging[j * (rowBytes / 2) + i] = toHalf(src[j * w + i] * scale);
    device.queue.writeTexture({ texture: tex }, staging, { bytesPerRow: rowBytes }, [w, h]);
    return tex;
  }
  const i16 = (u8) => new Int16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const sdfM = meta.sdf, hM = meta.height, fM = meta.far, cM = meta.landclass;
  const sdfTex = halfTexture(i16(data.sdf), sdfM.w, sdfM.h, sdfM.scale, 'river sdf');
  const hgtTex = halfTexture(i16(data.height), hM.w, hM.h, hM.scale, 'land height');
  const farTex = halfTexture(i16(data.far), fM.w, fM.h, fM.scale, 'far height');
  const clsTex = device.createTexture({ label: 'land class', size: [cM.w, cM.h], format: 'r8uint', usage: T.TEXTURE_BINDING | T.COPY_DST });
  {
    const rowBytes = Math.ceil(cM.w / 256) * 256;
    const st = new Uint8Array(rowBytes * cM.h);
    for (let j = 0; j < cM.h; j++) st.set(data.landclass.subarray(j * cM.w, (j + 1) * cM.w), j * rowBytes);
    device.queue.writeTexture({ texture: clsTex }, st, { bytesPerRow: rowBytes }, [cM.w, cM.h]);
  }
  // overhead occlusion (bridge decks over the water): r = underside height, g = soft coverage
  const occTex = device.createTexture({ label: 'overhead occlusion', size: [sdfM.w, sdfM.h], format: 'rg16float', usage: T.TEXTURE_BINDING | T.COPY_DST });
  {
    const w = sdfM.w, h = sdfM.h, cell = sdfM.cell;
    const under = new Float32Array(w * h), cov = new Float32Array(w * h);
    for (const o of geom.bridges.occluders) {
      const [ax, az] = o.a, [bx, bz] = o.b;
      const L = Math.hypot(bx - ax, bz - az), ux = (bx - ax) / L, uz = (bz - az) / L;
      for (let s = 0; s <= L; s += cell * 0.5) {
        const f = Math.min(o.profile.length - 1.001, s / o.profStep), i = Math.floor(f), t = f - i;
        const deck = o.profile[i] * (1 - t) + o.profile[i + 1] * t;
        for (let q = -o.width / 2; q <= o.width / 2; q += cell * 0.5) {
          const x = ax + ux * s - uz * q, z = az + uz * s + ux * q;
          const gi = Math.floor((x - sdfM.x0) / cell), gj = Math.floor((z - sdfM.z0) / cell);
          if (gi < 0 || gj < 0 || gi >= w || gj >= h) continue;
          const k = gj * w + gi;
          under[k] = Math.max(under[k], deck - o.depth);
          cov[k] = 1;
        }
      }
    }
    // soften coverage (light wraps in from the sides): box blur radius 3 cells, 2 passes
    const tmp = new Float32Array(w * h);
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        let s = 0, c = 0;
        for (let k = -3; k <= 3; k++) { const ii = i + k; if (ii >= 0 && ii < w) { s += cov[j * w + ii]; c++; } }
        tmp[j * w + i] = s / c;
      }
      for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
        let s = 0, c = 0;
        for (let k = -3; k <= 3; k++) { const jj = j + k; if (jj >= 0 && jj < h) { s += tmp[jj * w + i]; c++; } }
        cov[j * w + i] = s / c;
      }
    }
    const rowBytes = Math.ceil((w * 4) / 256) * 256;
    const st = new Uint16Array((rowBytes / 2) * h);
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      st[j * (rowBytes / 2) + i * 2] = toHalf(under[j * w + i]);
      st[j * (rowBytes / 2) + i * 2 + 1] = toHalf(cov[j * w + i]);
    }
    device.queue.writeTexture({ texture: occTex }, st, { bytesPerRow: rowBytes }, [w, h]);
  }

  // ------------------------------------------------------------------------------------------
  // compute pipelines
  // ------------------------------------------------------------------------------------------
  const [initPipe, evolvePipe, rowsPipe, colsPipe, simMipPipe, foamGenPipe, rgba8MipPipe, rgba16MipPipe,
    transPipe, msPipe, skyViewPipe, envPipe, cubeMipPipe, sunVisPipe, detailPipe, pathPipe, probePipe, exposurePipe] = await Promise.all([
    computePipe(simInit(N), 'sim init'), computePipe(simEvolve(N), 'sim evolve'), computePipe(fftRows(N), 'fft rows'),
    computePipe(fftCols(N), 'fft cols'), computePipe(SIM_MIPS, 'sim mips'), computePipe(FOAM_TEX, 'foam tex'),
    computePipe(RGBA8_MIPS, 'rgba8 mips'), computePipe(RGBA16_MIPS, 'rgba16 mips'),
    computePipe(TRANS_LUT, 'transmittance'), computePipe(MS_LUT, 'multiscatter'), computePipe(SKYVIEW_LUT, 'skyview'),
    computePipe(ENV_CUBE, 'env cube'), computePipe(CUBE_MIPS, 'cube mips'), computePipe(SUN_VIS, 'sun visibility'),
    computePipe(DETAIL_TEX, 'detail tex'), computePipe(PATH_FIELD, 'path field'), computePipe(HEIGHT_PROBE, 'probe'),
    computePipe(EXPOSURE, 'exposure'),
  ]);

  // ---- FFT simulation resources ----
  const C = CASCADES;
  const SIM_MIP_COUNT = Math.log2(N) + 1;
  const specBytes = N * N * C * 16;
  const simUBO = buf(112, U.UNIFORM | U.COPY_DST, 'sim ubo');
  const h0Buf = buf(specBytes, U.STORAGE, 'h0');
  const specA = buf(specBytes, U.STORAGE, 'specA'), specB = buf(specBytes, U.STORAGE, 'specB');
  const tmpA = buf(specBytes, U.STORAGE, 'tmpA'), tmpB = buf(specBytes, U.STORAGE, 'tmpB');
  const mkSimTex = (label) => device.createTexture({ label, size: [N, N, C], format: 'rgba16float', mipLevelCount: SIM_MIP_COUNT, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
  const dispTex = mkSimTex('disp'), derivTex = mkSimTex('deriv'), momTex = mkSimTex('moments');
  const bgInit = bgAuto(initPipe, [{ buffer: simUBO }, { buffer: h0Buf }]);
  const bgEvolve = bgAuto(evolvePipe, [{ buffer: simUBO }, { buffer: h0Buf }, { buffer: specA }, { buffer: specB }]);
  const bgRows = bgAuto(rowsPipe, [{ buffer: specA }, { buffer: specB }, { buffer: tmpA }, { buffer: tmpB }]);
  const bgCols = bgAuto(colsPipe, [{ buffer: tmpA }, { buffer: tmpB }, lvl(dispTex, 0), lvl(derivTex, 0), lvl(momTex, 0)]);
  const bgSimMips = [];
  for (let m = 1; m < SIM_MIP_COUNT; m++) bgSimMips.push(bgAuto(simMipPipe, [lvl(dispTex, m - 1), lvl(derivTex, m - 1), lvl(momTex, m - 1), lvl(dispTex, m), lvl(derivTex, m), lvl(momTex, m)]));

  // ---- procedural detail textures (generated once) ----
  const FOAM_SIZE = 512, FOAM_MIPS = 10;
  const mkDetail = (label) => device.createTexture({ label, size: [FOAM_SIZE, FOAM_SIZE], format: 'rgba8unorm', mipLevelCount: FOAM_MIPS, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
  const foamTex = mkDetail('foam detail'), detailTex = mkDetail('surface detail');
  {
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    for (const [pipe, tex] of [[foamGenPipe, foamTex], [detailPipe, detailTex]]) {
      cp.setPipeline(pipe);
      cp.setBindGroup(0, bgAuto(pipe, [lvl(tex, 0, '2d')]));
      cp.dispatchWorkgroups(FOAM_SIZE / 8, FOAM_SIZE / 8);
      cp.setPipeline(rgba8MipPipe);
      for (let m = 1; m < FOAM_MIPS; m++) {
        const s = FOAM_SIZE >> m;
        cp.setBindGroup(0, bgAuto(rgba8MipPipe, [lvl(tex, m - 1, '2d'), lvl(tex, m, '2d')]));
        cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8));
      }
    }
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  // ---- route field for the wake: (arc length, lateral offset, tangent) of the nearest route point ----
  const PF_CELL = 2;
  let pfx0 = Infinity, pfz0 = Infinity, pfx1 = -Infinity, pfz1 = -Infinity;
  const routePath = geom.path;
  for (let i = 0; i < routePath.x.length; i++) { pfx0 = Math.min(pfx0, routePath.x[i]); pfx1 = Math.max(pfx1, routePath.x[i]); pfz0 = Math.min(pfz0, routePath.z[i]); pfz1 = Math.max(pfz1, routePath.z[i]); }
  pfx0 -= 280; pfz0 -= 280; pfx1 += 280; pfz1 += 280;
  const PFW = Math.ceil((pfx1 - pfx0) / PF_CELL), PFH = Math.ceil((pfz1 - pfz0) / PF_CELL);
  const pathTex = device.createTexture({ label: 'route field', size: [PFW, PFH], format: 'rgba32float', usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
  {
    const pts = new Float32Array(routePath.x.length * 2);
    for (let i = 0; i < routePath.x.length; i++) { pts[i * 2] = routePath.x[i]; pts[i * 2 + 1] = routePath.z[i]; }
    const ptsBuf = upload(pts, U.STORAGE, 'route points');
    const pfU = upload(new Float32Array([pfx0, pfz0, PF_CELL, 0, routePath.x.length, routePath.step, 0, 0]), U.UNIFORM, 'route field ubo');
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(pathPipe);
    cp.setBindGroup(0, bgAuto(pathPipe, [{ buffer: pfU }, { buffer: ptsBuf }, pathTex.createView()]));
    cp.dispatchWorkgroups(Math.ceil(PFW / 8), Math.ceil(PFH / 8));
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  const passTimeBuf = buf(routePath.x.length * 4, U.STORAGE | U.COPY_DST, 'route pass times');

  // ---- atmosphere ----
  const atmoUBO = buf(16, U.UNIFORM | U.COPY_DST, 'atmo ubo');
  const mkLut = (w, h, label) => device.createTexture({ label, size: [w, h], format: 'rgba16float', usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
  const transLUT = mkLut(256, 64, 'transmittance'), msLUT = mkLut(32, 32, 'multiscatter'), skyLUT = mkLut(SKY_W, SKY_H, 'skyview');
  const envTex = device.createTexture({ label: 'env cube', size: [ENV_SIZE, ENV_SIZE, 6], format: 'rgba16float', mipLevelCount: ENV_MIPS, usage: T.TEXTURE_BINDING | T.STORAGE_BINDING });
  const envCubeView = envTex.createView({ dimension: 'cube' });
  const FRAME_BYTES = FRAME.floats * 4;
  const frameUBO = buf(FRAME_BYTES, U.UNIFORM | U.COPY_DST, 'frame ubo');
  const reflUBO = buf(FRAME_BYTES, U.UNIFORM | U.COPY_DST, 'refl frame ubo');
  const sunVisBuf = buf(16, U.STORAGE, 'sun visibility');
  const transView = transLUT.createView(), msView = msLUT.createView(), skyView = skyLUT.createView();
  const bgTrans = bgAuto(transPipe, [{ buffer: atmoUBO }, transView]);
  const bgMS = bgAuto(msPipe, [{ buffer: atmoUBO }, transView, sampClamp, msView]);
  const bgSkyView = bgAuto(skyViewPipe, [{ buffer: frameUBO }, transView, msView, sampClamp, skyView]);
  const bgEnv = bgAuto(envPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, lvl(envTex, 0)]);
  const bgSunVis = bgAuto(sunVisPipe, [{ buffer: frameUBO }, transView, skyView, sampClamp, { buffer: sunVisBuf }]);
  const bgCubeMips = [];
  for (let m = 1; m < ENV_MIPS; m++) bgCubeMips.push(bgAuto(cubeMipPipe, [lvl(envTex, m - 1), lvl(envTex, m)]));

  // ---- height probe ----
  const probeUArr = new Float32Array(28);
  const probeUBO = buf(probeUArr.byteLength, U.UNIFORM | U.COPY_DST, 'probe ubo');
  const probeOut = buf(16, U.STORAGE | U.COPY_SRC, 'probe out');
  const probeRead = buf(16, U.MAP_READ | U.COPY_DST, 'probe read');
  const bgProbe = bgAuto(probePipe, [{ buffer: probeUBO }, fullArr(dispTex), sampRepeat, { buffer: probeOut }]);
  let probePending = false;
  const waterH = new Float32Array(4);

  // ---- shadows ----
  const shadowTex = device.createTexture({ label: 'shadow cascades', size: [Q.shadow, Q.shadow, 3], format: 'depth32float', usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
  const shadowArrView = shadowTex.createView({ dimension: '2d-array' });
  const shadowLayerViews = [0, 1, 2].map((i) => shadowTex.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }));
  const shadowUBO = buf(256 * 3, U.UNIFORM | U.COPY_DST, 'shadow pass ubo');

  // ------------------------------------------------------------------------------------------
  // bind group layouts
  // ------------------------------------------------------------------------------------------
  const VF = S.VERTEX | S.FRAGMENT;
  const sceneBGL = device.createBindGroupLayout({
    label: 'scene', entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 2, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 3, visibility: VF, sampler: { type: 'filtering' } },
      { binding: 4, visibility: VF, texture: { sampleType: 'float', viewDimension: 'cube' } },
      { binding: 5, visibility: VF, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
      { binding: 6, visibility: VF, sampler: { type: 'comparison' } },
      { binding: 7, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 8, visibility: VF, sampler: { type: 'filtering' } },
      { binding: 9, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 10, visibility: VF, texture: { sampleType: 'float' } },
    ],
  });
  const sceneBG = (ubo) => device.createBindGroup({
    layout: sceneBGL, entries: [
      { binding: 0, resource: { buffer: ubo } }, { binding: 1, resource: transView }, { binding: 2, resource: skyView },
      { binding: 3, resource: sampClamp }, { binding: 4, resource: envCubeView }, { binding: 5, resource: shadowArrView },
      { binding: 6, resource: sampShadow }, { binding: 7, resource: { buffer: sunVisBuf } }, { binding: 8, resource: sampRepeat },
      { binding: 9, resource: occTex.createView() },
      { binding: 10, resource: sdfTex.createView() },
    ],
  });
  const bgSceneMain = sceneBG(frameUBO), bgSceneRefl = sceneBG(reflUBO);

  const meshBGL = device.createBindGroupLayout({
    label: 'mesh', entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform', hasDynamicOffset: true } },
      { binding: 1, visibility: VF, texture: { sampleType: 'float' } },
    ],
  });
  const objUBO = buf(256 * 2, U.UNIFORM | U.COPY_DST, 'object ubo');
  const bgMesh = device.createBindGroup({ layout: meshBGL, entries: [{ binding: 0, resource: { buffer: objUBO, size: 80 } }, { binding: 1, resource: detailTex.createView() }] });
  {
    const id = new Float32Array(20);
    id[0] = id[5] = id[10] = id[15] = 1;
    device.queue.writeBuffer(objUBO, 0, id);
  }
  const terrainBGL = device.createBindGroupLayout({
    label: 'terrain', entries: [
      { binding: 0, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 1, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 2, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 3, visibility: VF, texture: { sampleType: 'uint' } },
      { binding: 4, visibility: VF, texture: { sampleType: 'float' } },
    ],
  });
  const bgTerrain = device.createBindGroup({ layout: terrainBGL, entries: [sdfTex, hgtTex, farTex, clsTex, detailTex].map((t, i) => ({ binding: i, resource: t.createView() })) });
  const waterBGL = device.createBindGroupLayout({
    label: 'water', entries: [
      { binding: 0, visibility: VF, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 1, visibility: VF, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 2, visibility: VF, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 3, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 4, visibility: VF, texture: { sampleType: 'unfilterable-float' } },
      { binding: 5, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 6, visibility: VF, texture: { sampleType: 'float' } },
      { binding: 7, visibility: VF, sampler: { type: 'filtering' } },
      { binding: 8, visibility: VF, buffer: { type: 'read-only-storage' } },
    ],
  });
  const treesBGL = device.createBindGroupLayout({ label: 'trees', entries: [{ binding: 0, visibility: VF, texture: { sampleType: 'float' } }] });
  const bgTrees = device.createBindGroup({ layout: treesBGL, entries: [{ binding: 0, resource: detailTex.createView() }] });
  const shadowBGL = device.createBindGroupLayout({ label: 'shadow', entries: [{ binding: 0, visibility: S.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } }] });
  const bgShadow = device.createBindGroup({ layout: shadowBGL, entries: [{ binding: 0, resource: { buffer: shadowUBO, size: 80 } }] });
  const layout = (...bgls) => device.createPipelineLayout({ bindGroupLayouts: bgls });

  // ------------------------------------------------------------------------------------------
  // render pipelines (main: 4x MSAA; refl: 1x)
  // ------------------------------------------------------------------------------------------
  const [terrainMod, meshMod, waterMod, skyMod, skyCompMod, treesMod, spriteMod, meshShMod, treesShMod, downMod, upMod, compMod] = await Promise.all([
    shader(TERRAIN, 'terrain'), shader(MESH, 'mesh'), shader(WATER, 'water'), shader(SKY_BG, 'sky'), shader(SKY_COMPOSITE, 'sky composite'), shader(TREES, 'trees'),
    shader(SPRITES, 'sprites'), shader(MESH_SHADOW, 'mesh shadow'), shader(TREES_SHADOW, 'trees shadow'),
    shader(BLOOM_DOWN, 'bloom down'), shader(BLOOM_UP, 'bloom up'), shader(COMPOSITE, 'composite'),
  ]);
  const gridInst = [{ arrayStride: 16, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }] }];
  const meshVB = [{
    arrayStride: MESH_STRIDE, attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
      { shaderLocation: 2, offset: 16, format: 'unorm8x4' }, { shaderLocation: 3, offset: 20, format: 'uint8x4' },
      { shaderLocation: 4, offset: 24, format: 'float32x2' },
    ],
  }];
  const treeVB = [
    { arrayStride: TREE_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }, { shaderLocation: 2, offset: 24, format: 'float32' }] },
    { arrayStride: 20, stepMode: 'instance', attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x4' }, { shaderLocation: 4, offset: 16, format: 'float32' }] },
  ];
  const spriteVB = [{ arrayStride: 32, stepMode: 'instance', attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }, { shaderLocation: 1, offset: 16, format: 'float32x4' }] }];
  const additive = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' } };

  async function variants(desc) {
    const mk = (samples, label) => device.createRenderPipelineAsync({
      ...desc, label: desc.label + label,
      multisample: { count: samples },
      fragment: { ...desc.fragment, targets: [{ format: HDR, blend: desc.blend }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: desc.depthWrite ?? true, depthCompare: desc.depthCompare || 'greater' },
    });
    const [main, refl] = await Promise.all([mk(MSAA, ' main'), mk(1, ' refl')]);
    return { main, refl };
  }
  const P = {};
  const vehVB = [meshVB[0], { arrayStride: 32, stepMode: 'instance', attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x4' }, { shaderLocation: 6, offset: 16, format: 'float32x4' }] }];
  [P.terrain, P.mesh, P.trees, P.sky, P.sprites, P.vehicles] = await Promise.all([
    variants({ label: 'terrain', layout: layout(sceneBGL, terrainBGL), vertex: { module: terrainMod, entryPoint: 'vs', buffers: gridInst }, fragment: { module: terrainMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-list', cullMode: 'none' } }),
    variants({ label: 'mesh', layout: layout(sceneBGL, meshBGL), vertex: { module: meshMod, entryPoint: 'vs', buffers: meshVB }, fragment: { module: meshMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-list', cullMode: 'none' } }),
    variants({ label: 'trees', layout: layout(sceneBGL, treesBGL), vertex: { module: treesMod, entryPoint: 'vs', buffers: treeVB }, fragment: { module: treesMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-list', cullMode: 'none' } }),
    variants({ label: 'sky', layout: layout(sceneBGL), vertex: { module: skyMod, entryPoint: 'vs' }, fragment: { module: skyMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-list' }, depthWrite: false, depthCompare: 'equal' }),
    variants({ label: 'sprites', layout: layout(sceneBGL), vertex: { module: spriteMod, entryPoint: 'vs', buffers: spriteVB }, fragment: { module: spriteMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-strip' }, depthWrite: false, blend: additive }),
    variants({ label: 'vehicles', layout: layout(sceneBGL, meshBGL), vertex: { module: meshMod, entryPoint: 'vsInst', buffers: vehVB }, fragment: { module: meshMod, entryPoint: 'fs' }, primitive: { topology: 'triangle-list', cullMode: 'none' } }),
  ]);
  // half-resolution sky (clouds are soft; the full-res pass was ~3 ms) + composite into the main pass
  P.skyHalf = await device.createRenderPipelineAsync({
    label: 'sky half', layout: layout(sceneBGL), vertex: { module: skyMod, entryPoint: 'vs' },
    fragment: { module: skyMod, entryPoint: 'fs', targets: [{ format: HDR }] }, primitive: { topology: 'triangle-list' },
  });
  P.skyComp = await device.createRenderPipelineAsync({
    label: 'sky composite', layout: 'auto', vertex: { module: skyCompMod, entryPoint: 'vs' },
    fragment: { module: skyCompMod, entryPoint: 'fs', targets: [{ format: HDR }] }, primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'equal' }, multisample: { count: MSAA },
  });
  P.water = await device.createRenderPipelineAsync({
    label: 'water', layout: layout(sceneBGL, waterBGL),
    vertex: { module: waterMod, entryPoint: 'vs', buffers: gridInst },
    fragment: { module: waterMod, entryPoint: 'fs', targets: [{ format: HDR }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    multisample: { count: MSAA },
  });
  const shadowDS = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 };
  P.meshShadow = await device.createRenderPipelineAsync({
    label: 'mesh shadow', layout: layout(shadowBGL, meshBGL),
    vertex: { module: meshShMod, entryPoint: 'vs', buffers: [{ arrayStride: MESH_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shadowDS,
  });
  P.treesShadow = await device.createRenderPipelineAsync({
    label: 'trees shadow', layout: layout(shadowBGL),
    vertex: { module: treesShMod, entryPoint: 'vs', buffers: [{ arrayStride: TREE_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }, { arrayStride: 20, stepMode: 'instance', attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x4' }, { shaderLocation: 4, offset: 16, format: 'float32' }] }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shadowDS,
  });
  const downPipe = await device.createRenderPipelineAsync({ label: 'bloom down', layout: 'auto', vertex: { module: downMod, entryPoint: 'vs' }, fragment: { module: downMod, entryPoint: 'fs', targets: [{ format: HDR }] } });
  const upPipe = await device.createRenderPipelineAsync({
    label: 'bloom up', layout: 'auto', vertex: { module: upMod, entryPoint: 'vs' },
    fragment: { module: upMod, entryPoint: 'fs', targets: [{ format: HDR, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
  });
  const compPipe = await device.createRenderPipelineAsync({ label: 'composite', layout: 'auto', vertex: { module: compMod, entryPoint: 'vs' }, fragment: { module: compMod, entryPoint: 'fs', targets: [{ format }] } });

  // ------------------------------------------------------------------------------------------
  // geometry
  // ------------------------------------------------------------------------------------------
  // Static meshes are merged into one vertex + index buffer; each tile is one drawIndexed with its
  // own bounds for culling. The boat has its own buffers (it moves).
  const meshes = []; // { first, count, base, min, max, kind }
  let tris = 0;
  {
    const parts = [...geom.buildings.meshes.map((m) => [m, 'building']), [geom.bridges.mesh, 'bridge'], [geom.viaducts.mesh, 'bridge'], [geom.landmarks.mesh, 'landmark']];
    if (geom.misc) parts.push([geom.misc, 'misc']);
    let vbytes = 0, icount = 0;
    for (const [m] of parts) { vbytes += m.vertices.byteLength; icount += m.icount; }
    const vAll = new Uint8Array(vbytes), iAll = new Uint32Array(icount);
    let vo = 0, io = 0;
    for (const [m, kind] of parts) {
      if (!m.icount) continue;
      vAll.set(m.vertices, vo);
      iAll.set(m.indices.subarray(0, m.icount), io);
      meshes.push({ first: io, count: m.icount, base: vo / MESH_STRIDE, min: m.min, max: m.max, kind });
      vo += m.vertices.byteLength; io += m.icount;
      tris += m.icount / 3;
    }
    var staticVB = upload(vAll, U.VERTEX, 'static vertices');
    var staticIB = upload(iAll, U.INDEX, 'static indices');
  }
  const boatVB = upload(geom.boat.mesh.vertices, U.VERTEX, 'boat vertices');
  const boatIB = upload(geom.boat.mesh.indices, U.INDEX, 'boat indices');
  const boatCount = geom.boat.mesh.icount;
  // traffic: car / bus templates + per-frame instance buffers
  const vehicleKinds = ['car', 'bus'].map((k) => ({
    vb: upload(geom[k].vertices, U.VERTEX, k), ib: upload(geom[k].indices, U.INDEX, k), count: geom[k].icount,
    inst: buf(4096 * 32, U.VERTEX | U.COPY_DST, k + ' instances'), n: 0,
  }));
  log(`static meshes: ${meshes.length} tiles, ${(tris / 1e6).toFixed(2)} M triangles`);

  // trees
  const treeTpl = geom.treeTemplate;
  const treeVBuf = upload(treeTpl.vertices, U.VERTEX, 'tree template');
  const treeIBuf = upload(treeTpl.indices, U.INDEX, 'tree idx');
  const treeData = new Float32Array(data.trees.buffer, data.trees.byteOffset, data.trees.byteLength / 4);
  const treeCount = treeData.length / 5;
  // sort trees into 500 m cells so distant/off-screen cells can be skipped
  const treeCells = new Map();
  for (let i = 0; i < treeCount; i++) {
    const k = Math.floor(treeData[i * 5] / 500) + ',' + Math.floor(treeData[i * 5 + 1] / 500);
    if (!treeCells.has(k)) treeCells.set(k, []);
    treeCells.get(k).push(i);
  }
  const treeSorted = new Float32Array(treeCount * 5);
  const treeGroups = [];
  {
    let o = 0;
    for (const [k, list] of treeCells) {
      const [ci, cj] = k.split(',').map(Number);
      const start = o;
      for (const i of list) { treeSorted.set(treeData.subarray(i * 5, i * 5 + 5), o * 5); o++; }
      let ymax = 0;
      for (const i of list) ymax = Math.max(ymax, treeData[i * 5 + 2]);
      treeGroups.push({ start, count: list.length, min: [ci * 500, -2, cj * 500], max: [ci * 500 + 500, ymax + 15, cj * 500 + 500] });
    }
  }
  const treeInstBuf = upload(treeSorted, U.VERTEX, 'tree instances');

  // light sprites: static set (lamps, bridges, landmarks, aviation lights) + dynamic (boat, traffic)
  const lampData = new Float32Array(data.lamps.buffer, data.lamps.byteOffset, data.lamps.byteLength / 4);
  const staticLights = [];
  for (let i = 0; i < lampData.length; i += 3) staticLights.push(lampData[i], lampData[i + 1], lampData[i + 2], 0.3, 1.0, 0.78, 0.5, -10);
  const pushArr = (arr) => { for (let i = 0; i < arr.length; i++) staticLights.push(arr[i]); };
  pushArr(geom.bridges.lights);
  pushArr(geom.viaducts.lights);
  pushArr(geom.landmarks.lights);
  const av = geom.buildings.lights;
  for (let i = 0; i < av.length; i += 3) staticLights.push(av[i], av[i + 1], av[i + 2], 0.25, 1.0, 0.06, 0.03, -10);
  const staticSpriteBuf = upload(new Float32Array(staticLights), U.VERTEX, 'static lights');
  const staticSpriteCount = staticLights.length / 8;
  const MAX_DYN = 8192;
  const dynSpriteData = new Float32Array(MAX_DYN * 8);
  const dynSpriteBuf = buf(dynSpriteData.byteLength, U.VERTEX | U.COPY_DST, 'dynamic lights');
  let dynSpriteCount = 0;

  // CDLOD grids
  const GRID = 32;
  const gridIdx = upload(gridIndices(GRID), U.INDEX, 'grid idx');
  const waterLOD = new CDLOD({ leaf: 12, grid: GRID, lods: 11, maxDist: 14000, yMin: -2, yMax: 2, xyMargin: 2, maxNodes: 4096 });
  const landLOD = new CDLOD({ leaf: 48, grid: GRID, lods: 10, maxDist: 30000, yMin: -6, yMax: 650, xyMargin: 1, maxNodes: 4096 });
  const landLODRefl = new CDLOD({ leaf: 96, grid: GRID, lods: 9, maxDist: 30000, yMin: -6, yMax: 650, xyMargin: 1, maxNodes: 4096 });
  const waterInst = buf(waterLOD.maxNodes * 16, U.VERTEX | U.COPY_DST, 'water nodes');
  const landInst = buf(landLOD.maxNodes * 16, U.VERTEX | U.COPY_DST, 'land nodes');
  const landInstRefl = buf(landLOD.maxNodes * 16, U.VERTEX | U.COPY_DST, 'land nodes refl');

  // ------------------------------------------------------------------------------------------
  // screen-size targets
  // ------------------------------------------------------------------------------------------
  let W = 0, H = 0, RW = 0, RH = 0;
  let colorMS, depthMS, hdrTex, bloomTex, reflTex, reflDepth, skyTex, reflMips = 1;
  let bgSkyComp = null;
  const skyPass = { colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] };
  const skyUBO = buf(FRAME.floats * 4, U.UNIFORM | U.COPY_DST, 'sky frame ubo');
  const bgSceneSky = sceneBG(skyUBO);
  let downPasses = [], upPasses = [], downBGs = [], upBGs = [], bgComp = null, bgReflMips = [], bgExposure = null;
  const bloomUBOs = [];
  for (let i = 0; i < BLOOM_LEVELS * 2; i++) bloomUBOs.push(buf(16, U.UNIFORM | U.COPY_DST, 'bloom ubo'));
  const compUBO = buf(32, U.UNIFORM | U.COPY_DST, 'composite ubo');
  const compData = new Float32Array(8);
  const exposureBuf = buf(16, U.STORAGE | U.COPY_SRC, 'exposure');
  const exposureUBO = buf(16, U.UNIFORM | U.COPY_DST, 'exposure ubo');
  const exposureRead = buf(16, U.MAP_READ | U.COPY_DST, 'exposure read');
  let exposurePending = false;
  const exposureVal = new Float32Array(2);
  const mainPass = {
    colorAttachments: [{ view: null, resolveTarget: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  const reflPass = {
    colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 20000 }, loadOp: 'clear', storeOp: 'store' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  const compPass = { colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }], timestampWrites: undefined };
  const shadowPasses = shadowLayerViews.map((v) => ({ colorAttachments: [], depthStencilAttachment: { view: v, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } }));

  function resize(w, h) {
    if (w === W && h === H) return;
    W = w; H = h;
    for (const t of [colorMS, depthMS, hdrTex, bloomTex, reflTex, reflDepth, skyTex]) t && t.destroy();
    skyTex = device.createTexture({ size: [Math.max(1, W >> 1), Math.max(1, H >> 1)], format: HDR, usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    skyPass.colorAttachments[0].view = skyTex.createView();
    bgSkyComp = bgAuto(P.skyComp, [skyTex.createView(), linSamp]);
    colorMS = device.createTexture({ size: [W, H], format: HDR, sampleCount: MSAA, usage: T.RENDER_ATTACHMENT });
    depthMS = device.createTexture({ size: [W, H], format: 'depth32float', sampleCount: MSAA, usage: T.RENDER_ATTACHMENT });
    hdrTex = device.createTexture({ size: [W, H], format: HDR, usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.COPY_SRC });
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    bloomTex = device.createTexture({ size: [bw, bh], format: HDR, mipLevelCount: BLOOM_LEVELS, usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING });
    RW = Math.max(16, Math.round(W * Q.reflScale)); RH = Math.max(16, Math.round(H * Q.reflScale));
    reflMips = Math.min(6, Math.floor(Math.log2(Math.min(RW, RH))) - 2);
    reflTex = device.createTexture({ size: [RW, RH], format: HDR, mipLevelCount: reflMips, usage: T.RENDER_ATTACHMENT | T.TEXTURE_BINDING | T.STORAGE_BINDING });
    reflDepth = device.createTexture({ size: [RW, RH], format: 'depth32float', usage: T.RENDER_ATTACHMENT });
    mainPass.colorAttachments[0].view = colorMS.createView();
    mainPass.colorAttachments[0].resolveTarget = hdrTex.createView();
    mainPass.depthStencilAttachment.view = depthMS.createView();
    reflPass.colorAttachments[0].view = reflTex.createView({ baseMipLevel: 0, mipLevelCount: 1 });
    reflPass.depthStencilAttachment.view = reflDepth.createView();
    bgReflMips = [];
    for (let m = 1; m < reflMips; m++) bgReflMips.push(bgAuto(rgba16MipPipe, [reflTex.createView({ baseMipLevel: m - 1, mipLevelCount: 1 }), reflTex.createView({ baseMipLevel: m, mipLevelCount: 1 })]));
    const hdrView = hdrTex.createView();
    const bv = [];
    for (let m = 0; m < BLOOM_LEVELS; m++) bv.push(bloomTex.createView({ baseMipLevel: m, mipLevelCount: 1 }));
    downPasses = []; upPasses = []; downBGs = []; upBGs = [];
    const tmp = new Float32Array(4);
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const srcW = m === 0 ? W : Math.max(1, bw >> (m - 1)), srcH = m === 0 ? H : Math.max(1, bh >> (m - 1));
      tmp[0] = 1 / srcW; tmp[1] = 1 / srcH; tmp[2] = m === 0 ? 1 : 0; tmp[3] = 0;
      device.queue.writeBuffer(bloomUBOs[m], 0, tmp);
      downBGs.push(bgAuto(downPipe, [m === 0 ? hdrView : bv[m - 1], linSamp, { buffer: bloomUBOs[m] }]));
      downPasses.push({ colorAttachments: [{ view: bv[m], clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    }
    for (let m = BLOOM_LEVELS - 1; m > 0; m--) {
      const sw = Math.max(1, bw >> m), sh = Math.max(1, bh >> m);
      tmp[0] = 1 / sw; tmp[1] = 1 / sh; tmp[2] = 1.0; tmp[3] = 1.0;
      const ubo = bloomUBOs[BLOOM_LEVELS + m];
      device.queue.writeBuffer(ubo, 0, tmp);
      upBGs.push(bgAuto(upPipe, [bv[m], linSamp, { buffer: ubo }]));
      upPasses.push({ colorAttachments: [{ view: bv[m - 1], loadOp: 'load', storeOp: 'store' }] });
    }
    bgComp = bgAuto(compPipe, [hdrView, bv[0], linSamp, { buffer: compUBO }, { buffer: exposureBuf }]);
    bgExposure = bgAuto(exposurePipe, [bv[BLOOM_LEVELS - 1], { buffer: exposureUBO }, { buffer: exposureBuf }]);
  }
  const makeWaterBG = () => device.createBindGroup({
    layout: waterBGL, entries: [fullArr(dispTex), fullArr(derivTex), fullArr(momTex), foamTex.createView(), pathTex.createView(), reflTex.createView(), sdfTex.createView(), sampWater, { buffer: passTimeBuf }]
      .map((r, i) => ({ binding: i, resource: r })),
  });
  let bgWater = null, bgWaterW = -1;

  // ------------------------------------------------------------------------------------------
  // GPU timing
  // ------------------------------------------------------------------------------------------
  const hasTS = device.features.has('timestamp-query');
  let tsQuery = null, tsResolve = null, tsRead = null, tsPending = false;
  const timing = { gpuMs: 0 };
  let computeDesc = {};
  if (hasTS) {
    tsQuery = device.createQuerySet({ type: 'timestamp', count: 2 });
    tsResolve = buf(16, U.QUERY_RESOLVE | U.COPY_SRC, 'ts resolve');
    tsRead = buf(16, U.MAP_READ | U.COPY_DST, 'ts read');
    computeDesc = { timestampWrites: { querySet: tsQuery, beginningOfPassWriteIndex: 0 } };
    compPass.timestampWrites = { querySet: tsQuery, endOfPassWriteIndex: 1 };
  }

  // ------------------------------------------------------------------------------------------
  // per-frame uniforms
  // ------------------------------------------------------------------------------------------
  const frameF = new Float32Array(FRAME.floats);
  const reflF = new Float32Array(FRAME.floats);
  const skyF = new Float32Array(FRAME.floats);
  const proj = mat4(), view = mat4(), vp = mat4(), vpRefl = mat4(), tmpM = mat4(), tmpM2 = mat4();
  const shadowVP = [mat4(), mat4(), mat4()];      // as rendered (relative to shadowOrigin[c])
  const shadowVPUse = [mat4(), mat4(), mat4()];   // relative to the current camera, for lookups
  const shadowOrigin = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const shadowUArr = new Float32Array(64 * 3);
  const simF = new Float32Array(28);
  let needSpectrum = true, needLUT = true, envFull = 3;
  let frameIndex = 0;
  let lastWind = '';

  function writeSim(env, time, dt) {
    const f = simF;
    for (let c = 0; c < C; c++) {
      const L = CASC_L[c];
      f[c] = L;
      f[4 + c] = c === 0 ? Math.PI / L : (2 * Math.PI * 6) / L;
      f[8 + c] = c < C - 1 ? (2 * Math.PI * 6) / CASC_L[c + 1] : (Math.PI * N / L) * 0.98;
      f[12 + c] = [1.0, 1.0, 0.9, 0.7][c];
    }
    for (let c = 1; c < C; c++) f[4 + c] = f[8 + c - 1];
    f[16] = env.windSpeed; f[17] = Math.atan2(env.windDir[1], env.windDir[0]); f[18] = env.fetch; f[19] = 3.3;
    f[20] = env.choppy; f[21] = 0.0009; f[22] = LOOP; f[23] = 1;
    f[24] = time % LOOP; f[25] = dt; f[26] = 0; f[27] = 1.0;
    const key = `${env.windSpeed}|${env.windDir}|${env.fetch}|${env.choppy}`;
    if (key !== lastWind) { lastWind = key; needSpectrum = true; }
  }

  function setMat(f, off, m) { for (let i = 0; i < 16; i++) f[off + i] = m[i]; }
  function set4(f, name, a, b, c, d) { const o = FRAME[name]; f[o] = a; f[o + 1] = b; f[o + 2] = c; f[o + 3] = d; }

  function fillCommon(f, env, time) {
    set4(f, 'sunDir', env.sunDir[0], env.sunDir[1], env.sunDir[2], env.sunScale);
    set4(f, 'moonDir', env.moonDir[0], env.moonDir[1], env.moonDir[2], env.moonIllum);
    set4(f, 'cascL', ...CASC_L);
    set4(f, 'cascInvL', ...CASC_L.map((v) => 1 / v));
    set4(f, 'windP', env.windDir[0], env.windDir[1], env.windSpeed, env.gust);
    set4(f, 'current', env.current[0], env.current[1], time, 0);
    set4(f, 'lodP', waterLOD.r0, GRID, landLOD.r0, 0);
    set4(f, 'post', 1, env.bloom, frameIndex, env.debug || 0);
    set4(f, 'waterCol', env.waterCol[0], env.waterCol[1], env.waterCol[2], env.unresolved);
    set4(f, 'sssCol', env.sssCol[0], env.sssCol[1], env.sssCol[2], env.sss);
    set4(f, 'foamP', env.wakeFoam, env.wakeGain, 0, 0);
    const hz = env.haze;
    set4(f, 'fogExt', (5.802e-3 + 4.4e-3 * hz) * 1e-3, (13.558e-3 + 4.4e-3 * hz) * 1e-3, (33.1e-3 + 4.4e-3 * hz) * 1e-3, env.hazeFloor);
    set4(f, 'cloudP', 1.6, env.clouds, 2.2, time);
    set4(f, 'night', env.darkness, env.lights, env.skyglow, env.windowsLit);
    set4(f, 'shadowP', 1 / Q.shadow, SHADOW_FAR[0], SHADOW_FAR[1], SHADOW_FAR[2]);
    set4(f, 'sdfP', sdfM.x0, sdfM.z0, sdfM.w * sdfM.cell, sdfM.h * sdfM.cell);
    set4(f, 'hgtP', hM.x0, hM.z0, hM.w * hM.cell, hM.h * hM.cell);
    set4(f, 'farP', fM.x0, fM.z0, fM.w * fM.cell, fM.h * fM.cell);
    set4(f, 'misc', 1, env.shadows ? 1 : 0, reflMips, env.ambient);
    set4(f, 'screen', W, H, 1 / W, 1 / H);
  }

  // stable (texel-snapped) orthographic cascades around the camera
  const shadowUpdate = [true, true, true];
  let lastEnvObj = null;
  function buildShadows(cam, env) {
    const sd = env.sunDir;
    const L = [-sd[0], -sd[1], -sd[2]];
    // light basis
    let up = Math.abs(L[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0];
    const rx = norm3(cross3(up, L)), ry = cross3(L, rx);
    const tanY = Math.tan(cam.fovY / 2), tanX = tanY * W / H;
    let near = 0.05;
    for (let c = 0; c < 3; c++) {
      const far = SHADOW_FAR[c];
      if (!shadowUpdate[c]) {
        // cached cascade: re-express its matrix relative to the current camera position
        const o = shadowOrigin[c];
        translation(tmpM2, cam.pos[0] - o[0], cam.pos[1] - o[1], cam.pos[2] - o[2]);
        mul(shadowVPUse[c], shadowVP[c], tmpM2);
        near = far * 0.75;
        continue;
      }
      shadowOrigin[c] = cam.pos.slice();
      // bounding sphere of the frustum slice
      const mid = (near + far) / 2;
      const r = Math.hypot(far * tanX, far * tanY, (far - near) / 2) * 1.02;
      const cx = cam.fwd[0] * mid, cy = cam.fwd[1] * mid, cz = cam.fwd[2] * mid;
      const texel = (2 * r) / Q.shadow;
      // snap in a world-anchored light frame
      const wx = cx + cam.pos[0], wy = cy + cam.pos[1], wz = cz + cam.pos[2];
      let lx = rx[0] * wx + rx[1] * wy + rx[2] * wz, ly = ry[0] * wx + ry[1] * wy + ry[2] * wz;
      lx = Math.round(lx / texel) * texel; ly = Math.round(ly / texel) * texel;
      const camLx = rx[0] * cam.pos[0] + rx[1] * cam.pos[1] + rx[2] * cam.pos[2];
      const camLy = ry[0] * cam.pos[0] + ry[1] * cam.pos[1] + ry[2] * cam.pos[2];
      const ox = lx - camLx, oy = ly - camLy;
      const dc = L[0] * cx + L[1] * cy + L[2] * cz;
      const back = r + (c === 2 ? 900 : 650), front = r;
      const dmin = dc - back, dmax = dc + front;
      const m = shadowVP[c];
      m.fill(0);
      m[0] = rx[0] / r; m[4] = rx[1] / r; m[8] = rx[2] / r; m[12] = -ox / r;
      m[1] = ry[0] / r; m[5] = ry[1] / r; m[9] = ry[2] / r; m[13] = -oy / r;
      const iz = 1 / (dmax - dmin);
      m[2] = L[0] * iz; m[6] = L[1] * iz; m[10] = L[2] * iz; m[14] = -dmin * iz;
      m[15] = 1;
      shadowVPUse[c].set(m);
      near = far * 0.75;
    }
  }

  function writeFrames(cam, env, time) {
    const aspect = W / H;
    const tanY = Math.tan(cam.fovY * 0.5), tanX = tanY * aspect;
    perspective(proj, tanX, tanY, 0.05);
    viewFromBasis(view, cam.right, cam.up, cam.fwd);
    mul(vp, proj, view);
    // mirrored camera (about the mean water plane y = 0)
    const mr = [cam.right[0], -cam.right[1], cam.right[2]], mu = [cam.up[0], -cam.up[1], cam.up[2]], mf = [cam.fwd[0], -cam.fwd[1], cam.fwd[2]];
    viewFromBasis(tmpM, mr, mu, mf);
    mul(vpRefl, proj, tmpM);
    buildShadows(cam, env);

    const f = frameF;
    fillCommon(f, env, time);
    setMat(f, FRAME.viewProj, vp);
    // reflViewProj takes main-camera-relative points to the mirrored camera's clip space
    translation(tmpM2, 0, 2 * cam.pos[1], 0);
    mul(tmpM, vpRefl, tmpM2);
    setMat(f, FRAME.reflViewProj, tmpM);
    for (let c = 0; c < 3; c++) setMat(f, FRAME['shadowVP' + c], shadowVPUse[c]);
    set4(f, 'camPos', cam.pos[0], cam.pos[1], cam.pos[2], time);
    set4(f, 'camRight', cam.right[0] * tanX, cam.right[1] * tanX, cam.right[2] * tanX, 0);
    set4(f, 'camUp', cam.up[0] * tanY, cam.up[1] * tanY, cam.up[2] * tanY, 0);
    set4(f, 'camFwd', cam.fwd[0], cam.fwd[1], cam.fwd[2], 0);
    set4(f, 'atmo', env.haze, Math.max(cam.pos[1], 10) * 0.001, env.veil, 0.9);
    set4(f, 'boat', env.boat.x, env.boat.z, env.boat.heading, env.boat.v);
    set4(f, 'boat2', env.boat.yawRate, geom.boat.dims.length, geom.boat.dims.beam, geom.boat.dims.draft);
    set4(f, 'wakeP', env.boat.s, env.boat.wakeU, routePath.step, 0);
    set4(f, 'pathP', pfx0, pfz0, PF_CELL, 0);

    const g = reflF;
    g.set(f);
    setMat(g, FRAME.viewProj, vpRefl);
    translation(tmpM2, 0, -2 * cam.pos[1], 0);
    for (let c = 0; c < 3; c++) { mul(tmpM, shadowVPUse[c], tmpM2); setMat(g, FRAME['shadowVP' + c], tmpM); }
    set4(g, 'camPos', cam.pos[0], -cam.pos[1], cam.pos[2], time);
    set4(g, 'camRight', mr[0] * tanX, mr[1] * tanX, mr[2] * tanX, 0);
    set4(g, 'camUp', mu[0] * tanY, mu[1] * tanY, mu[2] * tanY, 0);
    set4(g, 'camFwd', mf[0], mf[1], mf[2], 1);
    set4(g, 'screen', RW, RH, 1 / RW, 1 / RH);
    device.queue.writeBuffer(frameUBO, 0, f);
    device.queue.writeBuffer(reflUBO, 0, g);
    // the half-res sky pass sees the same camera with half the pixels
    skyF.set(f);
    // keep the full-resolution pixel footprint: the cloud octaves must not fade earlier at half res
    set4(skyF, 'screen', W >> 1, H >> 1, 1 / W, 1 / H);
    device.queue.writeBuffer(skyUBO, 0, skyF);
    // shadow pass uniforms (camera-relative)
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 16; i++) shadowUArr[c * 64 + i] = shadowVP[c][i];
      shadowUArr[c * 64 + 16] = shadowOrigin[c][0]; shadowUArr[c * 64 + 17] = shadowOrigin[c][1]; shadowUArr[c * 64 + 18] = shadowOrigin[c][2];
    }
    device.queue.writeBuffer(shadowUBO, 0, shadowUArr);
  }

  // ---- culling ----
  const planes = new Float64Array(24);
  function extractPlanes(m, out, originX, originY, originZ, withNear) {
    // rows of the camera-relative matrix; planes in world space via origin shift
    const rows = [[3, 0, 1], [3, 0, -1], [3, 1, 1], [3, 1, -1]];
    let k = 0;
    for (const [a, b, s] of rows) {
      const px = m[a] + s * m[b], py = m[4 + a] + s * m[4 + b], pz = m[8 + a] + s * m[8 + b], pw = m[12 + a] + s * m[12 + b];
      out[k++] = px; out[k++] = py; out[k++] = pz; out[k++] = pw - (px * originX + py * originY + pz * originZ);
    }
    if (withNear) {
      // ortho: z in [0,1]
      let px = m[2], py = m[6], pz = m[10], pw = m[14];
      out[k++] = px; out[k++] = py; out[k++] = pz; out[k++] = pw - (px * originX + py * originY + pz * originZ);
      px = m[3] - m[2]; py = m[7] - m[6]; pz = m[11] - m[10]; pw = m[15] - m[14];
      out[k++] = px; out[k++] = py; out[k++] = pz; out[k++] = pw - (px * originX + py * originY + pz * originZ);
    }
    return k / 4;
  }
  function aabbVisible(P, n, min, max) {
    for (let p = 0; p < n * 4; p += 4) {
      const a = P[p], b = P[p + 1], c = P[p + 2], d = P[p + 3];
      const x = a > 0 ? max[0] : min[0], y = b > 0 ? max[1] : min[1], z = c > 0 ? max[2] : min[2];
      if (a * x + b * y + c * z + d < 0) return false;
    }
    return true;
  }
  const boatBounds = { min: [0, 0, 0], max: [0, 0, 0] };

  function drawMeshes(rp, pl, nPlanes, shadowPass, skipBoat, maxDist, camPos) {
    let n = 0;
    rp.setBindGroup(1, bgMesh, [0]);
    rp.setVertexBuffer(0, staticVB);
    rp.setIndexBuffer(staticIB, 'uint32');
    for (const m of meshes) {
      if (!aabbVisible(pl, nPlanes, m.min, m.max)) continue;
      if (maxDist && camPos && m.kind === 'building') {
        const dx = Math.max(m.min[0] - camPos[0], 0, camPos[0] - m.max[0]);
        const dz = Math.max(m.min[2] - camPos[2], 0, camPos[2] - m.max[2]);
        if (dx * dx + dz * dz > maxDist * maxDist) continue;
      }
      rp.drawIndexed(m.count, 1, m.first, m.base);
      n++;
    }
    if (!skipBoat && aabbVisible(pl, nPlanes, boatBounds.min, boatBounds.max)) {
      rp.setBindGroup(1, bgMesh, [256]);
      rp.setVertexBuffer(0, boatVB);
      rp.setIndexBuffer(boatIB, 'uint32');
      rp.drawIndexed(boatCount);
      n++;
    }
    return n;
  }
  function drawVehicles(rp, pipe) {
    rp.setPipeline(pipe);
    rp.setBindGroup(1, bgMesh, [0]);
    for (const k of vehicleKinds) {
      if (!k.n) continue;
      rp.setVertexBuffer(0, k.vb); rp.setVertexBuffer(1, k.inst); rp.setIndexBuffer(k.ib, 'uint32');
      rp.drawIndexed(k.count, k.n);
    }
  }
  function drawTrees(rp, pl, nPlanes, camPos, maxDist) {
    rp.setVertexBuffer(0, treeVBuf);
    rp.setVertexBuffer(1, treeInstBuf);
    rp.setIndexBuffer(treeIBuf, 'uint16');
    for (const g of treeGroups) {
      const dx = Math.max(g.min[0] - camPos[0], 0, camPos[0] - g.max[0]);
      const dz = Math.max(g.min[2] - camPos[2], 0, camPos[2] - g.max[2]);
      if (dx * dx + dz * dz > maxDist * maxDist) continue;
      if (!aabbVisible(pl, nPlanes, g.min, g.max)) continue;
      rp.drawIndexed(treeTpl.icount, g.count, 0, 0, g.start);
    }
  }

  // ------------------------------------------------------------------------------------------
  // frame
  // ------------------------------------------------------------------------------------------
  const objF = new Float32Array(20);
  let shotRequest = null;
  const stats = { gpuMs: 0, waterNodes: 0, landNodes: 0, meshesDrawn: 0, W: 0, H: 0, waterHeight: waterH, exposure: exposureVal };

  function render(state) {
    const { cam, env, time, dt, boatMatrix, dynLights } = state;
    const skip = env.skip || {};
    resize(state.width, state.height);
    if (bgWaterW !== W || !bgWater) { bgWater = makeWaterBG(); bgWaterW = W; }
    frameIndex++;

    // boat object transform + bounds (for culling)
    for (let i = 0; i < 16; i++) objF[i] = boatMatrix[i];
    objF[16] = 0.0006;
    device.queue.writeBuffer(objUBO, 256, objF);
    const bx = boatMatrix[12], by = boatMatrix[13], bz = boatMatrix[14];
    boatBounds.min = [bx - 30, by - 3, bz - 30]; boatBounds.max = [bx + 30, by + 15, bz + 30];

    if (state.passTimes) device.queue.writeBuffer(passTimeBuf, 0, state.passTimes);

    // near cascade every frame, middle every 2nd, far every 4th (static geometry, texel-snapped)
    const fullShadow = frameIndex < 4 || env !== lastEnvObj;
    lastEnvObj = env;
    shadowUpdate[0] = true; shadowUpdate[1] = fullShadow || frameIndex % 2 === 0; shadowUpdate[2] = fullShadow || frameIndex % 4 === 1;
    writeFrames(cam, env, time);
    writeSim(env, time, dt);
    device.queue.writeBuffer(simUBO, 0, simF);

    // CDLOD selections
    stats.waterNodes = waterLOD.select(cam.pos, vp);
    if (stats.waterNodes) device.queue.writeBuffer(waterInst, 0, waterLOD.data, 0, stats.waterNodes * 4);
    stats.landNodes = landLOD.select(cam.pos, vp);
    if (stats.landNodes) device.queue.writeBuffer(landInst, 0, landLOD.data, 0, stats.landNodes * 4);
    const reflCam = [cam.pos[0], -cam.pos[1], cam.pos[2]];
    const nLandRefl = landLODRefl.select(reflCam, vpRefl);
    if (nLandRefl) device.queue.writeBuffer(landInstRefl, 0, landLODRefl.data, 0, nLandRefl * 4);

    // probe points: bow, stern, port, starboard (world xz)
    for (let i = 0; i < 4; i++) { probeUArr[i * 4] = state.probe[i * 2]; probeUArr[i * 4 + 1] = state.probe[i * 2 + 1]; }
    for (let c = 0; c < 4; c++) { probeUArr[16 + c] = CASC_L[c]; probeUArr[20 + c] = 1 / CASC_L[c]; }
    probeUArr[24] = env.current[0]; probeUArr[25] = env.current[1]; probeUArr[26] = time;
    device.queue.writeBuffer(probeUBO, 0, probeUArr);

    // traffic bodies
    if (state.vehicles) {
      const v = state.vehicles;
      vehicleKinds[0].n = Math.min(4096, v.carCount); vehicleKinds[1].n = Math.min(4096, v.busCount);
      if (vehicleKinds[0].n) device.queue.writeBuffer(vehicleKinds[0].inst, 0, v.cars, 0, vehicleKinds[0].n * 8);
      if (vehicleKinds[1].n) device.queue.writeBuffer(vehicleKinds[1].inst, 0, v.buses, 0, vehicleKinds[1].n * 8);
    }
    // dynamic light sprites
    dynSpriteCount = Math.min(MAX_DYN, dynLights.count);
    if (dynSpriteCount) device.queue.writeBuffer(dynSpriteBuf, 0, dynLights.data, 0, dynSpriteCount * 8);

    const enc = device.createCommandEncoder();
    // ---------------- compute ----------------
    const cp = enc.beginComputePass(computeDesc);
    if (needSpectrum) {
      cp.setPipeline(initPipe); cp.setBindGroup(0, bgInit); cp.dispatchWorkgroups(N / 16, N / 16, C);
      needSpectrum = false;
    }
    if (needLUT || env.haze !== stats._haze) {
      stats._haze = env.haze;
      device.queue.writeBuffer(atmoUBO, 0, new Float32Array([env.haze, 0, 0, 0]));
      cp.setPipeline(transPipe); cp.setBindGroup(0, bgTrans); cp.dispatchWorkgroups(256 / 8, 64 / 8);
      cp.setPipeline(msPipe); cp.setBindGroup(0, bgMS); cp.dispatchWorkgroups(32 / 8, 32 / 8);
      needLUT = false; envFull = 3;
    }
    cp.setPipeline(evolvePipe); cp.setBindGroup(0, bgEvolve); cp.dispatchWorkgroups(N / 16, N / 16, C);
    cp.setPipeline(rowsPipe); cp.setBindGroup(0, bgRows); cp.dispatchWorkgroups(N, 1, C);
    cp.setPipeline(colsPipe); cp.setBindGroup(0, bgCols); cp.dispatchWorkgroups(N, 1, C);
    cp.setPipeline(simMipPipe);
    for (let m = 1; m < SIM_MIP_COUNT; m++) { const s = N >> m; cp.setBindGroup(0, bgSimMips[m - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), C); }

    cp.setPipeline(probePipe); cp.setBindGroup(0, bgProbe); cp.dispatchWorkgroups(1);
    cp.setPipeline(skyViewPipe); cp.setBindGroup(0, bgSkyView); cp.dispatchWorkgroups(Math.ceil(SKY_W / 8), Math.ceil(SKY_H / 8));
    cp.setPipeline(sunVisPipe); cp.setBindGroup(0, bgSunVis); cp.dispatchWorkgroups(1);
    cp.setPipeline(envPipe); cp.setBindGroup(0, bgEnv);
    cp.dispatchWorkgroups(ENV_SIZE / 8, ENV_SIZE / 8, envFull > 0 ? 6 : 1);
    if (envFull > 0) envFull--;
    cp.setPipeline(cubeMipPipe);
    for (let m = 1; m < ENV_MIPS; m++) { const s = Math.max(1, ENV_SIZE >> m); cp.setBindGroup(0, bgCubeMips[m - 1]); cp.dispatchWorkgroups(Math.ceil(s / 8), Math.ceil(s / 8), 6); }
    cp.end();

    // ---------------- shadows ----------------
    if (env.shadows && env.sunDir[1] > -0.01 && !skip.shadow) {
      for (let c = 0; c < 3; c++) {
        if (!shadowUpdate[c]) continue;
        const rp = enc.beginRenderPass(shadowPasses[c]);
        const np = extractPlanes(shadowVP[c], planes, cam.pos[0], cam.pos[1], cam.pos[2], false);
        rp.setPipeline(P.meshShadow);
        rp.setBindGroup(0, bgShadow, [c * 256]);
        drawMeshes(rp, planes, np, true, false);
        if (c < 2) {
          rp.setPipeline(P.treesShadow);
          rp.setBindGroup(0, bgShadow, [c * 256]);
          drawTrees(rp, planes, np, cam.pos, SHADOW_FAR[c] * 1.5);
        }
        rp.end();
      }
    }

    // ---------------- reflection ----------------
    if (!skip.refl) {
      const rp = enc.beginRenderPass(reflPass);
      const np = extractPlanes(vpRefl, planes, reflCam[0], reflCam[1], reflCam[2], false);
      rp.setBindGroup(0, bgSceneRefl);
      if (nLandRefl) {
        rp.setPipeline(P.terrain.refl); rp.setBindGroup(1, bgTerrain);
        rp.setVertexBuffer(0, landInstRefl); rp.setIndexBuffer(gridIdx, 'uint16');
        rp.drawIndexed(GRID * GRID * 6, nLandRefl);
      }
      rp.setPipeline(P.mesh.refl);
      drawMeshes(rp, planes, np, false, false, 2500, cam.pos);
      drawVehicles(rp, P.vehicles.refl);
      rp.setPipeline(P.trees.refl); rp.setBindGroup(1, bgTrees);
      drawTrees(rp, planes, np, cam.pos, Math.min(Q.treeDist, 1200));
      rp.setPipeline(P.sky.refl); rp.draw(3);
      if (env.lights > 0.001) {
        rp.setPipeline(P.sprites.refl);
        rp.setVertexBuffer(0, staticSpriteBuf); rp.draw(4, staticSpriteCount);
        if (dynSpriteCount) { rp.setVertexBuffer(0, dynSpriteBuf); rp.draw(4, dynSpriteCount); }
      }
      rp.end();
      const cpm = enc.beginComputePass();
      cpm.setPipeline(rgba16MipPipe);
      for (let m = 1; m < reflMips; m++) {
        const s = [Math.max(1, RW >> m), Math.max(1, RH >> m)];
        cpm.setBindGroup(0, bgReflMips[m - 1]);
        cpm.dispatchWorkgroups(Math.ceil(s[0] / 8), Math.ceil(s[1] / 8));
      }
      cpm.end();
    }

    // ---------------- sky (half resolution) ----------------
    if (!skip.sky) {
      const sp = enc.beginRenderPass(skyPass);
      sp.setPipeline(P.skyHalf); sp.setBindGroup(0, bgSceneSky); sp.draw(3);
      sp.end();
    }

    // ---------------- main ----------------
    {
      const rp = enc.beginRenderPass(mainPass);
      const np = extractPlanes(vp, planes, cam.pos[0], cam.pos[1], cam.pos[2], false);
      rp.setBindGroup(0, bgSceneMain);
      rp.setPipeline(P.mesh.main);
      if (!skip.meshes) stats.meshesDrawn = drawMeshes(rp, planes, np, false, false, 0, cam.pos);
      if (stats.landNodes && !skip.terrain) {
        rp.setPipeline(P.terrain.main); rp.setBindGroup(1, bgTerrain);
        rp.setVertexBuffer(0, landInst); rp.setIndexBuffer(gridIdx, 'uint16');
        rp.drawIndexed(GRID * GRID * 6, stats.landNodes);
      }
      if (!skip.meshes) drawVehicles(rp, P.vehicles.main);
      rp.setPipeline(P.trees.main); rp.setBindGroup(1, bgTrees);
      if (!skip.trees) drawTrees(rp, planes, np, cam.pos, Q.treeDist);
      if (stats.waterNodes && !skip.water) {
        rp.setPipeline(P.water); rp.setBindGroup(1, bgWater);
        rp.setVertexBuffer(0, waterInst); rp.setIndexBuffer(gridIdx, 'uint16');
        rp.drawIndexed(GRID * GRID * 6, stats.waterNodes);
      }
      if (!skip.sky) { rp.setPipeline(P.skyComp); rp.setBindGroup(0, bgSkyComp); rp.draw(3); rp.setBindGroup(0, bgSceneMain); }
      if (env.lights > 0.001) {
        rp.setPipeline(P.sprites.main);
        rp.setVertexBuffer(0, staticSpriteBuf); rp.draw(4, staticSpriteCount);
        if (dynSpriteCount) { rp.setVertexBuffer(0, dynSpriteBuf); rp.draw(4, dynSpriteCount); }
      }
      rp.end();
    }

    // ---------------- post ----------------
    for (let m = 0; m < BLOOM_LEVELS; m++) {
      const p = enc.beginRenderPass(downPasses[m]);
      p.setPipeline(downPipe); p.setBindGroup(0, downBGs[m]); p.draw(3); p.end();
    }
    for (let i = 0; i < upPasses.length; i++) {
      const p = enc.beginRenderPass(upPasses[i]);
      p.setPipeline(upPipe); p.setBindGroup(0, upBGs[i]); p.draw(3); p.end();
    }
    device.queue.writeBuffer(exposureUBO, 0, new Float32Array([state.resetExposure ? 0 : dt, env.exposureKey, env.exposureMin, env.exposureMax]));
    {
      const ep = enc.beginComputePass();
      ep.setPipeline(exposurePipe); ep.setBindGroup(0, bgExposure); ep.dispatchWorkgroups(1);
      ep.end();
    }
    compData[0] = env.exposure; compData[1] = env.bloom; compData[2] = frameIndex % 64; compData[3] = 0.06;
    compData[4] = env.saturation; compData[5] = env.contrast; compData[6] = 0; compData[7] = 0;
    device.queue.writeBuffer(compUBO, 0, compData);
    const swapTex = context.getCurrentTexture();
    compPass.colorAttachments[0].view = swapTex.createView();
    const pc = enc.beginRenderPass(compPass);
    pc.setPipeline(compPipe); pc.setBindGroup(0, bgComp); pc.draw(3); pc.end();

    let doProbe = false, doTs = false, doEx = false;
    if (!probePending && (frameIndex & 3) === 0) { enc.copyBufferToBuffer(probeOut, 0, probeRead, 0, 16); doProbe = true; }
    if (!exposurePending && (frameIndex & 7) === 0) { enc.copyBufferToBuffer(exposureBuf, 0, exposureRead, 0, 8); doEx = true; }
    if (hasTS && !tsPending && (frameIndex & 7) === 0) { enc.resolveQuerySet(tsQuery, 0, 2, tsResolve, 0); enc.copyBufferToBuffer(tsResolve, 0, tsRead, 0, 16); doTs = true; }
    let readBuf = null, bpr = 0;
    if (shotRequest) {
      bpr = Math.ceil(W * 4 / 256) * 256;
      readBuf = device.createBuffer({ size: bpr * H, usage: U.MAP_READ | U.COPY_DST });
      enc.copyTextureToBuffer({ texture: swapTex }, { buffer: readBuf, bytesPerRow: bpr }, [W, H]);
    }
    device.queue.submit([enc.finish()]);
    if (doProbe) { probePending = true; probeRead.mapAsync(GPUMapMode.READ).then(() => { waterH.set(new Float32Array(probeRead.getMappedRange(0, 16))); probeRead.unmap(); probePending = false; }, () => { probePending = false; }); }
    if (doEx) { exposurePending = true; exposureRead.mapAsync(GPUMapMode.READ).then(() => { exposureVal.set(new Float32Array(exposureRead.getMappedRange(0, 8))); exposureRead.unmap(); exposurePending = false; }, () => { exposurePending = false; }); }
    if (doTs) {
      tsPending = true;
      tsRead.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigUint64Array(tsRead.getMappedRange());
        const ms = Number(t[1] - t[0]) / 1e6;
        if (ms > 0 && ms < 1000) timing.gpuMs = timing.gpuMs ? timing.gpuMs * 0.8 + ms * 0.2 : ms;
        tsRead.unmap(); tsPending = false;
      }, () => { tsPending = false; });
    }
    if (readBuf) {
      const req = shotRequest; shotRequest = null;
      const w = W, h = H, fmt = format;
      readBuf.mapAsync(GPUMapMode.READ).then(() => {
        const src = new Uint8Array(readBuf.getMappedRange());
        const img = new ImageData(w, h);
        const bgra = fmt === 'bgra8unorm';
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const s = y * bpr + x * 4, d = (y * w + x) * 4;
          img.data[d] = src[s + (bgra ? 2 : 0)]; img.data[d + 1] = src[s + 1]; img.data[d + 2] = src[s + (bgra ? 0 : 2)]; img.data[d + 3] = 255;
        }
        readBuf.unmap(); readBuf.destroy();
        req(img);
      });
    }
    stats.gpuMs = timing.gpuMs; stats.W = W; stats.H = H;
  }

  return {
    render, stats, errors, quality: Q, hasTimestamps: hasTS,
    capture: () => new Promise((res) => { shotRequest = res; }),
    resetEnv: () => { envFull = 3; },
  };
}

function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
