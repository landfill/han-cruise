// Dynamic resolution and frame pacing for phones (touch-only devices).
//
// Resolution: the render size moves between a few fixed levels instead of a continuous scale.
// Every size change rebuilds about 85 MB of render targets at 1.4 MP (4x MSAA colour and depth,
// HDR, bloom, reflection, sky) and the canvas swap chain, which shows as a hitch on a phone, so
// the controller changes level only when it has to, holds a level for a while after a change and
// does not go back to a level that failed recently.
//
// Pacing: phones render on every n-th screen refresh, n chosen so the frame period is about
// 16 ms or more: every refresh at 60 Hz, every 2nd at 90 Hz (45 fps) and 120 Hz (60 fps), every
// 3rd at 144 Hz. A GPU that needs ~20 ms per frame on a 120 Hz screen otherwise shows an uneven
// mix of 8, 17 and 25 ms frames, and 60 fps on a 90 Hz screen alternates 11 and 22 ms. When even
// the lowest level cannot keep that rate, the app slows to every refresh nearest 33 ms (30 fps
// at 60 and 90 Hz), which is steadier than an uneven 35-45 fps and gives the GPU twice the time
// per frame, so the resolution can go back up (not on screens of 100 Hz and more, see below).
//
// Input, once per WINDOW: the smoothed GPU time when the browser has GPU timestamps; otherwise
// the time from submit to onSubmittedWorkDone (frames are paced, so the GPU is idle at submit
// unless it cannot keep up, and then the wait grows: an overestimate, which errs towards a lower
// level), plus the intervals between rendered frames. Pure logic, no DOM: tools/dynres-sim.mjs
// runs it offline.
export const LEVELS = [1, 0.9, 0.8, 0.7, 0.6];
export const WINDOW = 1.5;           // seconds between decisions
const HOLD_UP = 3;                   // seconds at a level before stepping up
const CEIL_TTL = 30;                 // a level that failed is not retried for this long
// GPU time aimed for, as a share of the frame period (13 ms at 60 fps, as before this change):
// headroom for the frame-to-frame variation (reflection every 2nd frame, shadow cascades)
const BUDGET = 0.78;
// slow down only when the lowest level would drop a large share of frames: a GPU at 18 ms per
// 16.7 ms period drops about 1 frame in 14 (better left alone), one at 25 ms every 2nd
const SLOW_AT = 1.3;
const LATE_DOWN = 0.05;              // share of frames a refresh late that counts as over budget
const SLOW_LATE = 0.25;              // ... and that slows the pace at the lowest level (fallback path)
const SLOW_RETRY = [20, 40, 80, 160];// seconds before retrying the normal pace (fallback path)
// No slow pace on screens of 100 Hz and more: their refreshes are so close together that a frame
// needing ~22-26 ms lands on different refreshes (25 or 33 ms on 120 Hz), so 30 fps wobbles about
// as much as the normal pace does, at a lower frame rate (tools/dynres-sim.mjs).
const SLOW_MIN_VSYNC = 10;

// refreshes per frame: normal pace (period >= ~16 ms) and slow pace (period nearest 33 ms)
export function refreshesPerFrame(pace, vsyncMs) {
  const v = vsyncMs || 1000 / 60;
  const normal = Math.max(1, Math.ceil(15.5 / v));
  return pace === 2 ? Math.max(normal + 1, Math.round(33.3 / v)) : normal;
}
export function periodMs(pace, vsyncMs) {
  const v = vsyncMs || 1000 / 60;
  return refreshesPerFrame(pace, v) * v;
}
function median(a) { const b = a.slice().sort((x, y) => x - y); return b[b.length >> 1]; }

export function createDynRes({ hasTimestamps }) {
  const st = {
    level: 0, pace: 1,
    changes: 0, paceChanges: 0,
    lastChange: -1e9,
    failedAt: LEVELS.map(() => -1e9),   // when each level last failed (at the current pace)
    calm: 0, slowCount: 0, normalFailedAt: -1e9,
    estimate: hasTimestamps ? 'timestamps' : 'done',
  };

  function setLevel(j, now) {
    j = Math.max(0, Math.min(LEVELS.length - 1, j));
    if (j !== st.level) { st.level = j; st.changes++; st.lastChange = now; }
    st.calm = 0;
  }
  function setPace(p, now) {
    if (p !== st.pace) { st.pace = p; st.paceChanges++; st.lastChange = now; st.failedAt.fill(-1e9); }
    st.calm = 0;
  }
  // predicted GPU time at level j from the time g at the current level (cost ~ pixel count,
  // which overstates the change, since part of the frame does not scale: errs on the safe side)
  const at = (j, g) => g * (LEVELS[j] / LEVELS[st.level]) ** 2;
  const FLOOR = LEVELS.length - 1;

  function updateGpu(g, now, v) {
    if (!g) return;
    const B = periodMs(st.pace, v) * BUDGET;
    if (g > B * 1.1) {
      st.failedAt[st.level] = now;
      if (st.pace === 1 && v >= SLOW_MIN_VSYNC && at(FLOOR, g) > periodMs(1, v) * SLOW_AT) {
        // even the lowest level misses the normal pace by far: slow down and pick the level for that
        const B2 = periodMs(2, v) * BUDGET;
        let k = 0;
        while (k < FLOOR && at(k, g) > B2) k++;
        setPace(2, now); setLevel(k, now);
        return;
      }
      // the largest level predicted to fit
      let j = Math.min(FLOOR, st.level + 1);
      while (j < FLOOR && at(j, g) > B) j++;
      setLevel(j, now);
      return;
    }
    if (g < B * 0.8) st.calm++; else st.calm = 0;
    if (st.calm < 2 || now - st.lastChange < HOLD_UP) return;
    // back to the normal pace when the lowest level fits it with a clear margin
    if (st.pace === 2) {
      const B1 = periodMs(1, v) * BUDGET;
      if (at(FLOOR, g) <= B1 * 0.85) {
        let k = 0;
        while (k < FLOOR && at(k, g) > B1 * 0.9) k++;
        setPace(1, now); setLevel(k, now);
        return;
      }
    }
    const up = st.level - 1;
    if (up >= 0 && at(up, g) <= B * 0.9 && now - st.failedAt[up] > CEIL_TTL) setLevel(up, now);
  }

  // share of rendered frames that came at least half a refresh after their paced time
  function lateShare(iv, v) {
    const late = periodMs(st.pace, v) + v * 0.5;
    let n = 0;
    for (const x of iv) if (x > late) n++;
    return iv.length ? n / iv.length : 0;
  }

  // fallback without timestamps and without submit->done samples: rendered-frame intervals only
  function updateFrames(iv, now, v) {
    if (iv.length < 10) return;
    const r = lateShare(iv, v);
    if (r > LATE_DOWN) {
      st.failedAt[st.level] = now;
      if (st.level === FLOOR) {
        if (st.pace === 1 && v >= SLOW_MIN_VSYNC && r > SLOW_LATE) { st.slowCount++; st.normalFailedAt = now; setPace(2, now); }
        st.calm = 0;
        return;
      }
      setLevel(st.level + (r > 0.3 ? 2 : 1), now);
      return;
    }
    if (r < 0.01) st.calm++; else st.calm = 0;
    if (st.calm < 3 || now - st.lastChange < HOLD_UP) return;
    if (st.level > 0 && now - st.failedAt[st.level - 1] > CEIL_TTL) { setLevel(st.level - 1, now); return; }
    // at the top level and calm at the slow pace: retry the normal pace, backing off after failures
    if (st.pace === 2 && st.level === 0) {
      const wait = SLOW_RETRY[Math.min(SLOW_RETRY.length - 1, Math.max(0, st.slowCount - 1))];
      if (now - st.normalFailedAt > wait) { setPace(1, now); setLevel(FLOOR, now); }
    }
  }

  return {
    state: st,
    get scale() { return LEVELS[st.level]; },
    get pace() { return st.pace; },
    // called once per WINDOW: gpuMs (smoothed, timestamps), intervals (ms between rendered
    // frames), doneMs (submit -> onSubmittedWorkDone per frame), vsyncMs (screen refresh interval)
    update({ gpuMs, intervals = [], doneMs, vsyncMs, now }) {
      const v = vsyncMs || 1000 / 60;
      if (hasTimestamps) { st.estimate = 'timestamps'; updateGpu(gpuMs, now, v); return; }
      if (doneMs && doneMs.length >= 10) {
        st.estimate = 'done';
        const g = median(doneMs);
        // frames that came a refresh late also count as over budget
        const over = intervals.length >= 10 && lateShare(intervals, v) > LATE_DOWN;
        updateGpu(over ? Math.max(g, periodMs(st.pace, v) * BUDGET * 1.11) : g, now, v);
        return;
      }
      st.estimate = 'frames';
      updateFrames(intervals, now, v);
    },
  };
}

// Frame limiter: should a rAF callback at `now` render? Yes once the paced period has passed,
// within half a refresh (rAF times jitter by a fraction of a millisecond).
export function paceDue(now, lastRender, pace, vsyncMs) {
  const v = vsyncMs || 1000 / 60;
  return now - lastRender >= periodMs(pace, v) - v * 0.5;
}
