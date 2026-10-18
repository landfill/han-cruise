// Solar and lunar direction for Seoul, from the NOAA solar position equations
// (accurate to a few hundredths of a degree, far below what is visible here).
// World frame: x = east, y = up, z = south.

const D2R = Math.PI / 180;
export const SEOUL = { lat: 37.525, lon: 126.945, tz: 9 };

function julianDay(date) { return date.getTime() / 86400000 + 2440587.5; }

// Returns { dir: [x,y,z], elevDeg, azimDeg } with azimuth clockwise from north.
export function sunPosition(date, lat = SEOUL.lat, lon = SEOUL.lon) {
  const jd = julianDay(date);
  const T = (jd - 2451545.0) / 36525;
  const L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360;
  const M = 357.52911 + T * (35999.05029 - 0.0001537 * T);
  const e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T);
  const C = Math.sin(M * D2R) * (1.914602 - T * (0.004817 + 0.000014 * T)) +
    Math.sin(2 * M * D2R) * (0.019993 - 0.000101 * T) + Math.sin(3 * M * D2R) * 0.000289;
  const trueLong = L0 + C;
  const omega = 125.04 - 1934.136 * T;
  const lambda = trueLong - 0.00569 - 0.00478 * Math.sin(omega * D2R);
  const eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60;
  const eps = eps0 + 0.00256 * Math.cos(omega * D2R);
  const decl = Math.asin(Math.sin(eps * D2R) * Math.sin(lambda * D2R));
  const y = Math.tan((eps / 2) * D2R) ** 2;
  const eqTime = 4 / D2R * (y * Math.sin(2 * L0 * D2R) - 2 * e * Math.sin(M * D2R) +
    4 * e * y * Math.sin(M * D2R) * Math.cos(2 * L0 * D2R) - 0.5 * y * y * Math.sin(4 * L0 * D2R) -
    1.25 * e * e * Math.sin(2 * M * D2R));
  const utcMin = date.getUTCHours() * 60 + date.getUTCMinutes() + date.getUTCSeconds() / 60;
  const tst = (utcMin + eqTime + 4 * lon) % 1440;
  const ha = (tst / 4 < 0 ? tst / 4 + 180 : tst / 4 - 180) * D2R;
  return fromHourAngle(ha, decl, lat);
}

function fromHourAngle(ha, decl, lat) {
  const phi = lat * D2R;
  const cosZen = Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(ha);
  const zen = Math.acos(Math.max(-1, Math.min(1, cosZen)));
  const elev = Math.PI / 2 - zen;
  // azimuth clockwise from north
  let az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(phi) - Math.tan(decl) * Math.cos(phi)) + Math.PI;
  az %= 2 * Math.PI;
  const ce = Math.cos(elev);
  return {
    dir: [Math.sin(az) * ce, Math.sin(elev), -Math.cos(az) * ce],
    elevDeg: elev / D2R,
    azimDeg: az / D2R,
  };
}

// Low-precision lunar position (Meeus ch. 47 leading terms, ~0.3 deg) plus illuminated fraction.
export function moonPosition(date, lat = SEOUL.lat, lon = SEOUL.lon) {
  const d = julianDay(date) - 2451545.0;
  const L = (218.316 + 13.176396 * d) * D2R;
  const M = (134.963 + 13.064993 * d) * D2R;
  const F = (93.272 + 13.22935 * d) * D2R;
  const lam = L + 6.289 * D2R * Math.sin(M);
  const bet = 5.128 * D2R * Math.sin(F);
  const eps = 23.4397 * D2R;
  const ra = Math.atan2(Math.sin(lam) * Math.cos(eps) - Math.tan(bet) * Math.sin(eps), Math.cos(lam));
  const decl = Math.asin(Math.sin(bet) * Math.cos(eps) + Math.cos(bet) * Math.sin(eps) * Math.sin(lam));
  const gmst = (280.16 + 360.9856235 * d) * D2R;
  const ha = gmst + lon * D2R - ra;
  const pos = fromHourAngle(ha, decl, lat);
  // phase: elongation from the sun
  const Ms = (357.5291 + 0.98560028 * d) * D2R;
  const Ls = Ms + (1.9148 * Math.sin(Ms) + 0.02 * Math.sin(2 * Ms) + 282.9372) * D2R + Math.PI;
  const elong = Math.acos(Math.cos(bet) * Math.cos(lam - Ls));
  pos.illum = (1 - Math.cos(elong)) / 2;
  return pos;
}

// Builds a Date for a Seoul local wall-clock time.
export function seoulDate(y, mo, d, h, mi) {
  return new Date(Date.UTC(y, mo - 1, d, h - SEOUL.tz, mi));
}
