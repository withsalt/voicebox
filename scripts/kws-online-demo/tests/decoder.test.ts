import { describe, expect, it } from "vitest";
import { KeywordDecoder } from "../src/dsp/decoder";

function posterior(token: number, probability = 0.95) {
  const p = new Float32Array(6).fill(0.001);
  p[token] = probability;
  return p;
}
describe("CTC keyword detection", () => {
  it("requires a blank between repeated symbols", () => {
    const decoder = new KeywordDecoder([1, 1]);
    for (let i = 0; i < 10; i++) expect(decoder.step(posterior(1))).toBeNull();
    expect(decoder.step(posterior(0))).toBeNull();
    expect(decoder.step(posterior(1))?.score).toBeCloseTo(0.95);
  });
  it("uses sqrt of token peak products and respects threshold changes", () => {
    const decoder = new KeywordDecoder([1, 2], 0.9);
    decoder.step(posterior(1, 0.8));
    expect(decoder.step(posterior(2, 0.8))).toBeNull();
    decoder.threshold = 0.7;
    expect(decoder.step(posterior(0))?.score).toBeCloseTo(0.8);
  });
  it("does not splice expired prefixes onto later speech", () => {
    const decoder = new KeywordDecoder([1, 2]);
    decoder.step(posterior(1));
    for (let i = 0; i < 120; i++) expect(decoder.step(posterior(0))).toBeNull();
    expect(decoder.step(posterior(2))).toBeNull();
  });
  it("keeps the absolute audio clock when resetting a completed search", () => {
    const decoder = new KeywordDecoder([1, 2]);
    decoder.step(posterior(1));
    expect(decoder.step(posterior(2))?.startFrame).toBe(0);
    decoder.reset();
    decoder.step(posterior(1));
    expect(decoder.step(posterior(2))?.startFrame).toBe(2);
  });
  it("does not report a keyword in prolonged silence", () => {
    const decoder = new KeywordDecoder([1, 2, 1, 2]);
    for (let i = 0; i < 5000; i++)
      expect(decoder.step(posterior(0))).toBeNull();
  });
});
