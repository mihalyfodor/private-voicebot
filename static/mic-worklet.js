// AudioWorkletProcessor that downsamples the mic input to 16 kHz mono Int16
// and posts fixed-size 512-sample frames (1024 bytes) to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate; // sampleRate is a worklet global
    this.srcPos = 0;           // fractional read position into the pending source buffer
    this.pending = new Float32Array(0); // leftover source-rate samples not yet consumed
    this.out = new Int16Array(512);
    this.outPos = 0;

    // Anti-alias filter for the decimation below: 2nd-order Butterworth low-pass at
    // 7 kHz (just under the 8 kHz Nyquist of the 16 kHz output), designed for this
    // context's sampleRate. Coefficients are the RBJ cookbook low-pass, normalized by
    // a0 so lowpass() is a plain difference equation.
    const fc = 7000;
    const Q = Math.SQRT1_2;          // 1/sqrt(2): maximally-flat (Butterworth) response
    const w0 = 2 * Math.PI * fc / sampleRate;
    const cosW0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Q);
    const a0 = 1 + alpha;
    this.b0 = (1 - cosW0) / 2 / a0;
    this.b1 = (1 - cosW0) / a0;
    this.b2 = this.b0;
    this.a1 = (-2 * cosW0) / a0;
    this.a2 = (1 - alpha) / a0;
    // Direct Form I state, persisted across process() calls: last two in/out samples.
    this.x1 = 0; this.x2 = 0; this.y1 = 0; this.y2 = 0;
  }

  lowpass(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x;
    this.y2 = this.y1; this.y1 = y;
    return y;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const channel = input[0];
    if (!channel || !channel.length) return true;

    // Leftover samples (already filtered) followed by this render quantum, filtered.
    const combined = new Float32Array(this.pending.length + channel.length);
    combined.set(this.pending, 0);
    for (let i = 0; i < channel.length; i++) combined[this.pending.length + i] = this.lowpass(channel[i]);

    let pos = this.srcPos;
    while (pos + this.ratio <= combined.length) {
      // Simple decimation with linear interpolation between neighboring samples.
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const a = combined[idx];
      const b = idx + 1 < combined.length ? combined[idx + 1] : a;
      const sample = a + (b - a) * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.out[this.outPos++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.outPos === this.out.length) {
        this.port.postMessage(this.out.buffer, [this.out.buffer]);
        this.out = new Int16Array(512);
        this.outPos = 0;
      }
      pos += this.ratio;
    }

    // Keep whatever's left (consumed samples fall behind `pos` rounded down).
    const consumedTo = Math.floor(pos);
    this.pending = combined.slice(consumedTo);
    this.srcPos = pos - consumedTo;

    return true;
  }
}

registerProcessor('mic-processor', MicProcessor);
