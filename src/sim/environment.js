// Time-of-day presets and the environment block passed to the renderer each frame.
// Sun and moon come from the real astronomical position for Seoul on the chosen date/time.
import { sunPosition, moonPosition, seoulDate } from './sun.js';
import { smoothstep } from '../gpu/math.js';

export const PRESETS = {
  day: { label: '낮', date: [2026, 10, 18, 13, 10], haze: 2.2, clouds: 0.0, veil: 0.16, wind: 3.2, key: 0.15, exMax: 20 },
  golden: { label: '골든아워', date: [2026, 10, 18, 17, 18], haze: 3.0, clouds: 0.0, veil: 0.40, wind: 2.8, key: 0.13, exMax: 40 },
  // night: a camera keeps the scene dark and lets lit windows and lamps carry the image
  night: { label: '야간', date: [2026, 10, 18, 19, 40], haze: 3.4, clouds: 0.0, veil: 0.30, wind: 2.4, key: 0.035, exMax: 160 },
};

// The river flows west-north-west past Yeouido (towards the Yellow Sea).
const DOWNSTREAM = [-0.86, -0.51];

export function makeEnv(presetKey, overrides = {}) {
  const p = PRESETS[presetKey];
  const date = seoulDate(...p.date);
  const sun = sunPosition(date);
  const moon = moonPosition(date);
  const e = sun.elevDeg;
  const env = {
    preset: presetKey, label: p.label, date, sunElev: e, sunAzim: sun.azimDeg,
    sunDir: sun.dir, sunScale: 14.0,
    // lunar illuminance relative to the sun's scale: full moon ~2.5e-6 of sunlight
    moonDir: moon.dir, moonIllum: 14.0 * 2.5e-6 * moon.illum * smoothstep(2, -6, e),
    // wind from the north-west (typical autumn monsoon); waves travel south-east
    windDir: [0.62, 0.78], windSpeed: p.wind, fetch: 2500, choppy: 0.85, gust: 1.0,
    current: [DOWNSTREAM[0] * 0.28, DOWNSTREAM[1] * 0.28],
    // turbid Han River: silt-laden olive green, low clarity
    waterCol: [0.030, 0.040, 0.027], unresolved: 0.0011,
    sssCol: [0.05, 0.075, 0.04], sss: 0.9,
    wakeFoam: 1.0, wakeGain: 0.38,
    haze: p.haze, hazeFloor: 0.9e-4, clouds: p.clouds, veil: p.veil,
    // night: artificial light scale (applies to windows, lamps, traffic), skyglow radiance
    darkness: smoothstep(4, -10, e),
    lights: smoothstep(6, -4, e),
    skyglow: 0.00035 * smoothstep(2, -12, e),
    windowsLit: 0.2 + 0.25 * smoothstep(6, -8, e),
    shadows: e > -1,
    ambient: 1.0,
    exposure: 1.0, exposureKey: p.key, exposureMin: 0.02, exposureMax: p.exMax,
    bloom: 0.018, saturation: 1.0, contrast: 1.0,
    debug: 0,
    boat: { x: 0, z: 0, heading: 0, v: 0, yawRate: 0 },
  };
  return Object.assign(env, overrides);
}
