import { describe, expect, it } from "vitest";
import golden from "./golden/frontend.json";
import { Frontend, KaldiFbank } from "../src/dsp/frontend";

const pcm = Float32Array.from(
  { length: golden.sampleCount },
  (_, i) =>
    0.15 * Math.sin((2 * Math.PI * 431 * i) / 16000) +
    0.07 * Math.cos((2 * Math.PI * 1277 * i) / 16000),
);
const cmvn = {
  add_shift: Array.from({ length: 400 }, (_, i) => (i % 7) - 3),
  rescale: Array.from({ length: 400 }, (_, i) =>
    Math.fround(0.1 + (i % 11) * 0.02),
  ),
};
function maxDifference(actual: ArrayLike<number>, expected: ArrayLike<number>) {
  expect(actual.length).toBe(expected.length);
  return Math.max(...Array.from(actual, (n, i) => Math.abs(n - expected[i])));
}

describe("Kaldi frontend against independently generated NumPy FFT reference", () => {
  it("matches the 80-bin power-spectrum Fbank", () => {
    expect(
      maxDifference(
        new KaldiFbank().compute(pcm.subarray(0, 400)),
        golden.firstFbank,
      ),
    ).toBeLessThan(2e-5);
  });
  it("preserves LFR left padding, frame count, and CMVN across arbitrary PCM packets", () => {
    const whole = new Frontend(cmvn).push(pcm);
    const streaming = new Frontend(cmvn),
      chunks: Float32Array[] = [];
    for (let offset = 0; offset < pcm.length; offset += 137)
      chunks.push(...streaming.push(pcm.subarray(offset, offset + 137)));
    expect(whole).toHaveLength(golden.lfrCount);
    expect(chunks).toHaveLength(whole.length);
    expect(maxDifference(chunks[0], golden.firstLfr)).toBeLessThan(2e-5);
    expect(maxDifference(chunks.at(-1)!, golden.lastLfr)).toBeLessThan(2e-5);
    chunks.forEach((chunk, i) =>
      expect(maxDifference(chunk, whole[i])).toBe(0),
    );
  });
  it("produces finite silence features without a false energy floor", () => {
    const features = new KaldiFbank().compute(new Float32Array(400));
    expect(
      [...features].every((n) => Math.abs(n - Math.log(1.1920929e-7)) < 1e-5),
    ).toBe(true);
  });
});
