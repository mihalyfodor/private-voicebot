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
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return true;
    const channel = input[0];
    if (!channel || !channel.length) return true;

    // Concatenate leftover samples with the new render quantum.
    const combined = new Float32Array(this.pending.length + channel.length);
    combined.set(this.pending, 0);
    combined.set(channel, this.pending.length);

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
