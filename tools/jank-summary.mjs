// Summarises a paired tools/jank.mjs run (labels "old ..." / "new ...", same config, several
// rounds) into a markdown table of medians over the rounds.
// Usage: node tools/jank-summary.mjs captures/perf/jank-ab.json
import fs from 'node:fs';

const runs = JSON.parse(fs.readFileSync(process.argv[2] || 'captures/perf/jank-ab.json', 'utf8'));
const med = (a) => { const b = a.slice().sort((x, y) => x - y); const n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const groups = {};
for (const r of runs) {
  const [side, ...cfg] = r.label.split(' ');
  const key = cfg.filter((x) => !/^r\d+$/.test(x)).join(' ');
  ((groups[key] ||= { old: [], new: [] })[side] || []).push(r);
}
const METRICS = [
  ['rAF 콜백 p50 (ms)', (r) => r.rafCallbackMs.p50, 2],
  ['rAF 콜백 p95 (ms)', (r) => r.rafCallbackMs.p95, 2],
  ['rAF 콜백 p99 (ms)', (r) => r.rafCallbackMs.p99, 2],
  ['rAF 콜백 최대 (ms)', (r) => r.rafCallbackMs.max, 1],
  ['프레임 간격 p99 (ms)', (r) => r.p99, 1],
  ['25 ms 초과 프레임 (%)', (r) => r.over25pct, 2],
  ['50 ms 초과 히치 (회/분)', (r) => r.hitches50 / (r.seconds / 60), 1],
  ['롱태스크 (회/분)', (r) => r.longTasks / (r.seconds / 60), 1],
  ['GC 횟수 (회/분)', (r) => (r.gc.minor + r.gc.major) / (r.seconds / 60), 0],
  ['GC 합계 (ms/분)', (r) => r.gc.totalMs / (r.seconds / 60), 0],
  ['GC 최대 (ms)', (r) => r.gc.maxMs, 1],
  ['메인 스레드 rAF 콜백 (ms/s)', (r) => r.mainThreadMsPerSec['rAF callback'] || 0, 0],
  ['캔버스 크기 변경 (회)', (r) => r.canvasResizes, 0],
  ['WebGPU 호출/프레임', (r) => r.gpuCallsPerFrame, 0],
];
const lines = [];
for (const [key, g] of Object.entries(groups)) {
  lines.push(`\n**${key}** (이전 ${g.old.length}회 / 이후 ${g.new.length}회, 중앙값)\n`);
  lines.push('| 지표 | 이전 | 이후 | 변화 |', '|---|---|---|---|');
  for (const [name, f, d] of METRICS) {
    const a = med(g.old.map(f)), b = med(g.new.map(f));
    const ch = a ? `${b >= a ? '+' : ''}${((b / a - 1) * 100).toFixed(0)}%` : '—';
    lines.push(`| ${name} | ${a.toFixed(d)} | ${b.toFixed(d)} | ${ch} |`);
  }
}
console.log(lines.join('\n'));
