import { describe, expect, it } from "vitest";
import { Resampler } from "../src/dsp/resampler";

function tone(rate: number, hz: number) {
  return Float32Array.from({ length: rate }, (_, i) =>
    Math.sin((2 * Math.PI * hz * i) / rate),
  );
}
function rms(pcm: Float32Array) {
  const stable = pcm.subarray(200, pcm.length - 200);
  return Math.sqrt(stable.reduce((n, v) => n + v * v, 0) / stable.length);
}

describe("streaming sinc resampler", () => {
  for (const rate of [44100, 48000]) {
    it(`retains phase across packet boundaries at ${rate} Hz`, () => {
      const pcm = tone(rate, 1000),
        expected = new Resampler(rate).push(pcm),
        stream = new Resampler(rate);
      const actual: number[] = [];
      for (let start = 0; start < pcm.length; start += 317)
        actual.push(...stream.push(pcm.subarray(start, start + 317)));
      expect(actual.length).toBe(expected.length);
      expect(
        Math.max(...actual.map((n, i) => Math.abs(n - expected[i]))),
      ).toBeLessThan(1e-7);
      expect(rms(expected)).toBeGreaterThan(0.7);
      expect(rms(expected)).toBeLessThan(0.71);
    });
    it(`rejects high-frequency aliasing when downsampling from ${rate} Hz`, () => {
      expect(rms(new Resampler(rate).push(tone(rate, 12000)))).toBeLessThan(
        0.003,
      );
    });
  }
  it("passes native 16 kHz input without changing samples", () => {
    const pcm = tone(16000, 1000);
    expect(new Resampler(16000).push(pcm)).toEqual(pcm);
  });
});
