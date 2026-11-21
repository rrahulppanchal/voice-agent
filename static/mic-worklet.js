// Captures mic audio, resamples to 16 kHz mono int16, posts 20 ms frames (320 samples).
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0; // fractional read position into `carry`
    this.carry = new Float32Array(0);
    this.out = new Int16Array(320);
    this.outLen = 0;
    this.levelAcc = 0;
    this.levelN = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;

    // join leftover samples with new block
    const buf = new Float32Array(this.carry.length + ch.length);
    buf.set(this.carry);
    buf.set(ch, this.carry.length);

    let p = this.pos;
    while (p + 1 < buf.length) {
      const i = Math.floor(p);
      const f = p - i;
      const s = buf[i] * (1 - f) + buf[i + 1] * f; // linear interpolation
      const v = Math.max(-1, Math.min(1, s));
      this.out[this.outLen++] = v < 0 ? v * 0x8000 : v * 0x7fff;
      this.levelAcc += v * v;
      this.levelN++;
      if (this.outLen === this.out.length) {
        const frame = this.out.slice(0);
        const rms = Math.sqrt(this.levelAcc / this.levelN);
        this.port.postMessage({ frame: frame.buffer, rms }, [frame.buffer]);
        this.outLen = 0;
        this.levelAcc = 0;
        this.levelN = 0;
      }
      p += this.ratio;
    }
    const keep = Math.floor(p);
    this.carry = buf.slice(keep);
    this.pos = p - keep;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
