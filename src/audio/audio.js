// Procedural soundscape (Web Audio, no recorded assets):
//   engine   - a slow-turning marine diesel heard through the deck: low firing harmonics + muffled noise
//   water    - water along the hull: band-passed noise whose level follows boat speed, with slow swells
//   wind     - apparent wind across the deck, low-passed and gusty
//   city     - distant traffic and city hum, louder and more reverberant under a bridge deck
// Starts only after a user gesture; everything is a few gain nodes so muting is instant.

function noiseBuffer(ctx, seconds, brown = false) {
  const n = Math.floor(ctx.sampleRate * seconds);
  const b = ctx.createBuffer(1, n, ctx.sampleRate);
  const d = b.getChannelData(0);
  let last = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.random() * 2 - 1;
    if (brown) { last = (last + 0.02 * w) / 1.02; d[i] = last * 3.5; } else d[i] = w;
  }
  return b;
}

export class CruiseAudio {
  constructor() { this.ctx = null; this.on = false; this.t = 0; }

  async start() {
    if (!this.ctx) this.build();
    if (this.ctx.state !== 'running') await this.ctx.resume();
    this.on = true;
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setTargetAtTime(0.9, now, 0.4);
  }
  stop() {
    if (!this.ctx) return;
    this.on = false;
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setTargetAtTime(0, now, 0.08);
  }

  build() {
    const ctx = this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'playback' });
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18; comp.ratio.value = 3;
    this.master.connect(comp).connect(ctx.destination);
    const white = noiseBuffer(ctx, 4), brown = noiseBuffer(ctx, 6, true);
    const loop = (buf) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, Math.random() * buf.duration); return s; };

    // ---- engine: 4-stroke diesel firing at ~ rpm/60 * cylinders/2; heard through the hull ----
    this.engine = ctx.createGain(); this.engine.gain.value = 0.0;
    const eLP = ctx.createBiquadFilter(); eLP.type = 'lowpass'; eLP.frequency.value = 260; eLP.Q.value = 0.7;
    this.engineOsc = [];
    for (const [mult, amp] of [[1, 0.5], [2, 0.32], [3, 0.16], [4.02, 0.08]]) {
      const o = ctx.createOscillator(); o.type = mult === 1 ? 'sawtooth' : 'triangle';
      const g = ctx.createGain(); g.gain.value = amp;
      o.connect(g).connect(eLP); o.start();
      this.engineOsc.push({ o, mult });
    }
    // combustion roughness: brown noise amplitude-modulated at the firing rate
    const rough = loop(brown); const rG = ctx.createGain(); rG.gain.value = 0.35;
    this.roughLFO = ctx.createOscillator(); this.roughLFO.frequency.value = 18;
    const lfoG = ctx.createGain(); lfoG.gain.value = 0.25;
    this.roughLFO.connect(lfoG).connect(rG.gain); this.roughLFO.start();
    rough.connect(rG).connect(eLP);
    eLP.connect(this.engine).connect(this.master);

    // ---- water along the hull ----
    this.water = ctx.createGain(); this.water.gain.value = 0;
    const wBP = ctx.createBiquadFilter(); wBP.type = 'bandpass'; wBP.frequency.value = 700; wBP.Q.value = 0.6;
    const wLP = ctx.createBiquadFilter(); wLP.type = 'lowpass'; wLP.frequency.value = 2600;
    loop(white).connect(wBP).connect(wLP).connect(this.water);
    // low gurgle / slap component
    const slap = ctx.createBiquadFilter(); slap.type = 'bandpass'; slap.frequency.value = 180; slap.Q.value = 1.2;
    this.slapG = ctx.createGain(); this.slapG.gain.value = 0.0;
    loop(brown).connect(slap).connect(this.slapG).connect(this.master);
    const wPan = ctx.createStereoPanner(); wPan.pan.value = 0;
    this.water.connect(wPan).connect(this.master);

    // ---- wind ----
    this.wind = ctx.createGain(); this.wind.gain.value = 0;
    const wiLP = ctx.createBiquadFilter(); wiLP.type = 'lowpass'; wiLP.frequency.value = 420; wiLP.Q.value = 0.4;
    this.windLP = wiLP;
    loop(white).connect(wiLP).connect(this.wind).connect(this.master);

    // ---- distant city ----
    this.city = ctx.createGain(); this.city.gain.value = 0;
    const cLP = ctx.createBiquadFilter(); cLP.type = 'lowpass'; cLP.frequency.value = 520;
    const cHP = ctx.createBiquadFilter(); cHP.type = 'highpass'; cHP.frequency.value = 45;
    loop(brown).connect(cHP).connect(cLP).connect(this.city);
    // under-bridge reverb: a short synthetic impulse response (concrete deck, open sides)
    const ir = ctx.createBuffer(2, Math.floor(ctx.sampleRate * 1.6), ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3.2);
    }
    this.verb = ctx.createConvolver(); this.verb.buffer = ir;
    this.verbSend = ctx.createGain(); this.verbSend.gain.value = 0;
    this.city.connect(this.master);
    this.city.connect(this.verbSend);
    this.engine.connect(this.verbSend);
    this.water.connect(this.verbSend);
    this.verbSend.connect(this.verb).connect(this.master);
    // traffic over the bridge deck (louder rumble when directly underneath)
    this.deck = ctx.createGain(); this.deck.gain.value = 0;
    const dBP = ctx.createBiquadFilter(); dBP.type = 'bandpass'; dBP.frequency.value = 95; dBP.Q.value = 0.8;
    loop(brown).connect(dBP).connect(this.deck).connect(this.master);
    this.deck.connect(this.verbSend);
  }

  // RMS level of the mix (for automated checks)
  level() {
    if (!this.ctx) return 0;
    if (!this.analyser) { this.analyser = this.ctx.createAnalyser(); this.analyser.fftSize = 2048; this.master.connect(this.analyser); this.buf = new Float32Array(2048); }
    this.analyser.getFloatTimeDomainData(this.buf);
    let s = 0;
    for (const v of this.buf) s += v * v;
    return Math.sqrt(s / this.buf.length);
  }

  update(dt, s) {
    if (!this.ctx || !this.on) return;
    this.t += dt;
    const now = this.ctx.currentTime;
    const v = s.paused ? s.speed : s.speed;   // the boat drifts to a stop when paused
    const k = Math.min(1, v / 4.6);
    // engine: ~650 rpm idle -> ~1500 rpm cruise, 6 cylinders -> firing = rpm/60*3
    const rpm = 650 + 850 * k;
    const fire = rpm / 60 * 3;
    for (const e of this.engineOsc) e.o.frequency.setTargetAtTime(fire * e.mult / 2, now, 0.3);
    this.roughLFO.frequency.setTargetAtTime(fire / 2, now, 0.3);
    this.engine.gain.setTargetAtTime(0.10 + 0.12 * k, now, 0.4);
    // water: hiss with slow swells (bow wave rhythm), quiet at the berth
    const swell = 0.75 + 0.25 * Math.sin(this.t * 0.9) * Math.sin(this.t * 0.37 + 1.3);
    this.water.gain.setTargetAtTime((0.02 + 0.16 * k * k) * swell, now, 0.25);
    this.slapG.gain.setTargetAtTime(0.05 + 0.05 * (1 - k) * (0.5 + 0.5 * Math.sin(this.t * 1.7)), now, 0.2);
    // apparent wind across the deck
    const gust = 0.7 + 0.3 * Math.sin(this.t * 0.23) * Math.sin(this.t * 0.61 + 2.0);
    this.wind.gain.setTargetAtTime((0.035 + 0.07 * k) * gust, now, 0.5);
    this.windLP.frequency.setTargetAtTime(300 + 400 * k * gust, now, 0.5);
    // city and bridges
    const ub = s.underBridge || 0;
    this.city.gain.setTargetAtTime(0.05 + 0.03 * (1 - (s.night || 0)) + 0.05 * ub, now, 0.6);
    this.deck.gain.setTargetAtTime(0.32 * ub, now, 0.3);
    this.verbSend.gain.setTargetAtTime(0.5 * ub, now, 0.3);
  }
}
