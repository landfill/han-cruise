// Route sections for the HUD, derived from where the path actually crosses the bridge axes.
export function bridgeCrossings(path, bridges, key) {
  const br = bridges.find((b) => b.key === key);
  if (!br) return [];
  const [ax, az] = br.a, [bx, bz] = br.b;
  const L = Math.hypot(bx - ax, bz - az), ux = (bx - ax) / L, uz = (bz - az) / L;
  const side = (x, z) => (x - ax) * uz - (z - az) * ux;
  const out = [];
  for (let i = 1; i < path.x.length; i++) {
    const s0 = side(path.x[i - 1], path.z[i - 1]), s1 = side(path.x[i], path.z[i]);
    if (Math.sign(s0) === Math.sign(s1)) continue;
    const t = s0 / (s0 - s1);
    const x = path.x[i - 1] + (path.x[i] - path.x[i - 1]) * t, z = path.z[i - 1] + (path.z[i] - path.z[i - 1]) * t;
    const along = (x - ax) * ux + (z - az) * uz;
    if (along >= 0 && along <= L) out.push((i - 1 + t) * path.step);
  }
  return out;
}

export function routeSections(path, bridges) {
  const [w1, w2] = bridgeCrossings(path, bridges, 'wonhyo');
  const [r1, r2] = bridgeCrossings(path, bridges, 'rail_a');
  const L = path.length;
  return [
    { s0: 0, s1: 160, name: '여의도 선착장 출항' },
    { s0: 160, s1: w1 - 110, name: '원효대교 접근' },
    { s0: w1 - 110, s1: w1 + 90, name: '원효대교 통과' },
    { s0: w1 + 90, s1: r1 - 110, name: '원효대교 → 한강철교' },
    { s0: r1 - 110, s1: r1 + 90, name: '한강철교 통과' },
    { s0: r1 + 90, s1: r2 - 110, name: '노들섬 앞 선회' },
    { s0: r2 - 110, s1: r2 + 90, name: '귀항 · 한강철교 통과' },
    { s0: r2 + 90, s1: w2 - 110, name: '귀항 · 이촌 한강공원 앞' },
    { s0: w2 - 110, s1: w2 + 90, name: '귀항 · 원효대교 통과' },
    { s0: w2 + 90, s1: L - 260, name: '귀항 · 마포대교 방향' },
    { s0: L - 260, s1: L + 1, name: '여의도 선착장 입항' },
  ].map((x) => ({ ...x, crossings: { w1, w2, r1, r2 } }));
}
