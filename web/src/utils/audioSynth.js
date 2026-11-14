// Web Audio API procedural sound engine for F1 V10 Sound & Telemetry UI
let audioCtx = null;
let engineOsc1 = null;
let engineOsc2 = null;
let engineGain = null;
let isEngineRunning = false;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

export const playTelemetryBeep = (freq = 880, type = 'sine', duration = 0.08) => {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, ctx.currentTime + duration);

    gain.gain.setValueAtTime(0.06, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    // Audio context may require user gesture
  }
};

export const playGearShift = () => {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const now = ctx.currentTime;
    
    // Quick pneumatic pop / blowoff
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(450, now);
    osc.frequency.exponentialRampToValueAtTime(120, now + 0.12);
    
    gain.gain.setValueAtTime(0.12, now);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(now + 0.12);
  } catch (e) {}
};

export const startV10Engine = () => {
  try {
    const ctx = getAudioContext();
    if (!ctx || isEngineRunning) return;

    engineGain = ctx.createGain();
    engineGain.gain.setValueAtTime(0.05, ctx.currentTime);

    // Fundamental V10 idle tone (~90Hz = ~5400 RPM idle)
    engineOsc1 = ctx.createOscillator();
    engineOsc1.type = 'sawtooth';
    engineOsc1.frequency.setValueAtTime(95, ctx.currentTime);

    // 2nd harmonic screaming V10 tone
    engineOsc2 = ctx.createOscillator();
    engineOsc2.type = 'triangle';
    engineOsc2.frequency.setValueAtTime(190, ctx.currentTime);

    engineOsc1.connect(engineGain);
    engineOsc2.connect(engineGain);
    engineGain.connect(ctx.destination);

    engineOsc1.start();
    engineOsc2.start();
    isEngineRunning = true;
  } catch (e) {}
};

export const stopV10Engine = () => {
  try {
    if (!isEngineRunning || !engineGain || !audioCtx) return;
    engineGain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.3);
    setTimeout(() => {
      if (engineOsc1) { engineOsc1.stop(); engineOsc1.disconnect(); }
      if (engineOsc2) { engineOsc2.stop(); engineOsc2.disconnect(); }
      isEngineRunning = false;
    }, 350);
  } catch (e) {
    isEngineRunning = false;
  }
};

export const revEngine = (targetRPM = 18500) => {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;

    // Trigger high-rev scream
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const subOsc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(1800, now);
    filter.Q.setValueAtTime(3.0, now);

    const baseFreq = (targetRPM / 60) * (10 / 2); // V10 firing frequency: 10 cylinders / 2 revs
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(baseFreq * 0.4, now);
    osc.frequency.exponentialRampToValueAtTime(baseFreq, now + 0.5);
    osc.frequency.exponentialRampToValueAtTime(baseFreq * 0.6, now + 1.2);

    subOsc.type = 'triangle';
    subOsc.frequency.setValueAtTime(baseFreq * 0.8, now);
    subOsc.frequency.exponentialRampToValueAtTime(baseFreq * 2.0, now + 0.5);
    subOsc.frequency.exponentialRampToValueAtTime(baseFreq * 1.2, now + 1.2);

    gain.gain.setValueAtTime(0.001, now);
    gain.gain.linearRampToValueAtTime(0.15, now + 0.4);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.3);

    osc.connect(filter);
    subOsc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    subOsc.start(now);
    osc.stop(now + 1.35);
    subOsc.stop(now + 1.35);
  } catch (e) {}
};
