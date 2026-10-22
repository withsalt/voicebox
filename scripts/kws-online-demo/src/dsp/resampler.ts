/** Streaming windowed-sinc low-pass resampler. Keeps phase and FIR context across packets. */
export class Resampler {
  private buffer = new Float32Array(0);
  private base = 0;
  private outputs = 0;
  private readonly ratio: number;
  private readonly radius = 32;
  private readonly kernels = new Map<number, Float64Array>();

  constructor(
    readonly inputRate: number,
    readonly outputRate = 16000,
  ) {
    if (!Number.isFinite(inputRate) || inputRate < 8000 || inputRate > 192000)
      throw new Error("不支持此音频采样率。");
    this.ratio = inputRate / outputRate;
  }

  push(samples: Float32Array): Float32Array {
    if (this.inputRate === this.outputRate) return samples;
    const joined = new Float32Array(this.buffer.length + samples.length);
    joined.set(this.buffer);
    joined.set(samples, this.buffer.length);
    this.buffer = joined;
    const end = this.base + joined.length;
    const values: number[] = [];
    while (this.outputs * this.ratio + this.radius < end) {
      const position = this.outputs * this.ratio;
      const center = Math.floor(position);
      const fraction = position - center;
      const key = Math.round(fraction * 1e6);
      let kernel = this.kernels.get(key);
      if (!kernel) {
        const cutoff = Math.min(1, 1 / this.ratio) * 0.94;
        kernel = new Float64Array(this.radius * 2);
        let sum = 0;
        for (let j = 0; j < kernel.length; j++) {
          const d = j - this.radius + 1 - fraction;
          const sinc =
            Math.abs(d) < 1e-9
              ? cutoff
              : Math.sin(Math.PI * cutoff * d) / (Math.PI * d);
          const window =
            Math.abs(d) <= this.radius
              ? 0.5 * (1 + Math.cos((Math.PI * d) / this.radius))
              : 0;
          sum += kernel[j] = sinc * window;
        }
        for (let j = 0; j < kernel.length; j++) kernel[j] /= sum;
        if (this.kernels.size < 256) this.kernels.set(key, kernel);
      }
      let value = 0;
      for (let j = 0; j < kernel.length; j++) {
        const index = center - this.radius + 1 + j;
        if (index >= 0) value += (joined[index - this.base] || 0) * kernel[j];
      }
      values.push(value);
      this.outputs++;
    }
    const keepFrom = Math.max(
      this.base,
      Math.floor(this.outputs * this.ratio) - this.radius,
    );
    this.buffer = joined.slice(keepFrom - this.base);
    this.base = keepFrom;
    return Float32Array.from(values);
  }
}
