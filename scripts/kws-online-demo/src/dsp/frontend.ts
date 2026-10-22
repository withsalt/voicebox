import type { ModelMetadata } from "../lib/metadata";

const f32 = Math.fround;
const mel = (hz: number) => 1127 * Math.log(1 + hz / 700);

/** Kaldi-compatible 80-bin power Fbank, matching VoiceBox's FSMN frontend. */
export class KaldiFbank {
  private readonly window = Float32Array.from(
    { length: 400 },
    (_, i) => 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / 399),
  );
  private readonly filters: { start: number; weights: Float64Array }[];
  private readonly re = new Float64Array(512);
  private readonly im = new Float64Array(512);
  private readonly power = new Float64Array(256);

  constructor() {
    const low = mel(20),
      delta = (mel(8000) - low) / 81;
    this.filters = Array.from({ length: 80 }, (_, b) => {
      const left = low + b * delta,
        center = left + delta,
        right = center + delta;
      const indices: number[] = [],
        weights: number[] = [];
      for (let i = 0; i < 256; i++) {
        const m = mel((i * 16000) / 512);
        if (m > left && m < right) {
          indices.push(i);
          weights.push(Math.min((m - left) / delta, (right - m) / delta));
        }
      }
      return { start: indices[0] || 0, weights: Float64Array.from(weights) };
    });
  }

  compute(pcm: Float32Array): Float32Array {
    const re = this.re,
      im = this.im;
    re.fill(0);
    im.fill(0);
    let mean = 0;
    for (let i = 0; i < 400; i++) {
      re[i] = pcm[i] * 32768;
      mean += re[i];
    }
    mean /= 400;
    for (let i = 0; i < 400; i++) re[i] -= mean;
    for (let i = 399; i >= 1; i--)
      re[i] = (re[i] - 0.97 * re[i - 1]) * this.window[i];
    re[0] *= 0.03 * this.window[0];
    let j = 0;
    for (let i = 1; i < 512; i++) {
      let bit = 256;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) [re[i], re[j]] = [re[j], re[i]];
    }
    for (let length = 2; length <= 512; length *= 2) {
      const half = length / 2;
      for (let k = 0; k < half; k++) {
        const wr = Math.cos((-2 * Math.PI * k) / length),
          wi = Math.sin((-2 * Math.PI * k) / length);
        for (let i = k; i < 512; i += length) {
          const other = i + half;
          const tr = re[other] * wr - im[other] * wi,
            ti = re[other] * wi + im[other] * wr;
          re[other] = re[i] - tr;
          im[other] = im[i] - ti;
          re[i] += tr;
          im[i] += ti;
        }
      }
    }
    for (let i = 0; i < 256; i++) this.power[i] = re[i] ** 2 + im[i] ** 2;
    return Float32Array.from(this.filters, ({ start, weights }) => {
      let sum = 0;
      for (let i = 0; i < weights.length; i++)
        sum += weights[i] * this.power[start + i];
      return Math.log(Math.max(sum, 1.1920929e-7));
    });
  }
}

export class Frontend {
  private readonly fbank = new KaldiFbank();
  private samples = new Float32Array(0);
  private readonly history = new Map<number, Float32Array>();
  private first: Float32Array | null = null;
  private frames = 0;
  private next = 0;
  constructor(private readonly cmvn: ModelMetadata["cmvn"]) {}

  push(pcm: Float32Array): Float32Array[] {
    const joined = new Float32Array(this.samples.length + pcm.length);
    joined.set(this.samples);
    joined.set(pcm, this.samples.length);
    const output: Float32Array[] = [];
    let offset = 0;
    for (; offset + 400 <= joined.length; offset += 160) {
      const frame = this.fbank.compute(joined.subarray(offset, offset + 400));
      if (!this.first) this.first = frame;
      this.history.set(this.frames++, frame);
      while (this.next * 3 + 2 < this.frames) {
        const stacked = new Float32Array(400);
        for (let j = 0; j < 5; j++) {
          const index = this.next * 3 - 2 + j;
          stacked.set(
            index < 0 ? this.first : this.history.get(index)!,
            j * 80,
          );
        }
        for (let d = 0; d < 400; d++)
          stacked[d] = f32(
            f32(stacked[d] + this.cmvn.add_shift[d]) * this.cmvn.rescale[d],
          );
        output.push(stacked);
        this.next++;
        for (const index of this.history.keys())
          if (index < this.next * 3 - 2) this.history.delete(index);
      }
    }
    this.samples = joined.slice(offset);
    return output;
  }
}
