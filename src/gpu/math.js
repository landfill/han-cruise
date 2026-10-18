// Column-major 4x4 matrix helpers (Float64Array for precision on the CPU side).
export const mat4 = () => new Float64Array(16);

export function mul(out, a, b) {
  const t = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    t[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    t[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    t[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    t[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  out.set(t);
  return out;
}

// Reverse-Z infinite perspective (depth 1 at the near plane, 0 at infinity)
export function perspective(out, tanX, tanY, near) {
  out.fill(0);
  out[0] = 1 / tanX; out[5] = 1 / tanY; out[11] = -1; out[14] = near;
  return out;
}

// View rotation from camera basis (right, up, forward); translation excluded (camera-relative)
export function viewFromBasis(out, r, u, f) {
  out.fill(0);
  out[0] = r[0]; out[4] = r[1]; out[8] = r[2];
  out[1] = u[0]; out[5] = u[1]; out[9] = u[2];
  out[2] = -f[0]; out[6] = -f[1]; out[10] = -f[2];
  out[15] = 1;
  return out;
}

export function translation(out, x, y, z) {
  out.fill(0);
  out[0] = out[5] = out[10] = out[15] = 1;
  out[12] = x; out[13] = y; out[14] = z;
  return out;
}

// Rigid transform: rotation from yaw (cw from north, about +y), pitch (about local x, bow up +),
// roll (about local -z forward, starboard down +), then translation.
export function boatMatrix(out, x, y, z, heading, pitch, roll) {
  // local axes: right = +x, up = +y, back = +z (forward is -z)
  const ch = Math.cos(heading), sh = Math.sin(heading);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const cr = Math.cos(roll), sr = Math.sin(roll);
  // R = Ryaw * Rpitch * Rroll
  // Ryaw maps local (1,0,0)->(ch,0,sh), (0,0,-1)->(sh,0,-ch)
  const Ry = [ch, 0, sh, 0, 1, 0, -sh, 0, ch];           // columns: x, y, z
  const Rx = [1, 0, 0, 0, cp, sp, 0, -sp, cp];           // pitch about x (bow up when +)
  const Rz = [cr, -sr, 0, sr, cr, 0, 0, 0, 1];           // roll about z
  const m3 = (A, B) => {
    const o = new Array(9);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) o[c * 3 + r] = A[r] * B[c * 3] + A[3 + r] * B[c * 3 + 1] + A[6 + r] * B[c * 3 + 2];
    return o;
  };
  const R = m3(m3(Ry, Rx), Rz);
  out.fill(0);
  out[0] = R[0]; out[1] = R[1]; out[2] = R[2];
  out[4] = R[3]; out[5] = R[4]; out[6] = R[5];
  out[8] = R[6]; out[9] = R[7]; out[10] = R[8];
  out[12] = x; out[13] = y; out[14] = z; out[15] = 1;
  return out;
}

export function transformPoint(m, p) {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}
export function transformDir(m, p) {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2], m[1] * p[0] + m[5] * p[1] + m[9] * p[2], m[2] * p[0] + m[6] * p[1] + m[10] * p[2]];
}

export const D2R = Math.PI / 180;
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// IEEE half-float conversion
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
export function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const s = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return s;
    m = (m | 0x800000) >> (1 - e);
    return s | ((m + 0x1000) >> 13);
  }
  if (e >= 31) return s | 0x7c00;
  const h = s | (e << 10) | ((m + 0x1000) >> 13);
  return h;
}
