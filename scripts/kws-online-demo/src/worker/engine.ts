import * as ort from "onnxruntime-web/wasm";
import type { ModelMetadata } from "../lib/metadata";
import { Frontend } from "../dsp/frontend";
import { Resampler } from "../dsp/resampler";
import { KeywordDecoder } from "../dsp/decoder";

export interface Wakeup {
  keyword: string;
  score: number;
  startMs: number;
  endMs: number;
}
export interface AudioResult {
  detections: Wakeup[];
  audioSeconds: number;
  warming: boolean;
  inferenceMs: number;
}

export class KwsEngine {
  private session: ort.InferenceSession | null = null;
  private states: Record<string, ort.Tensor> = {};
  private metadata!: ModelMetadata;
  private frontend!: Frontend;
  private resampler!: Resampler;
  private decoder!: KeywordDecoder;
  private features: Float32Array[] = [];
  private samples = 0;
  private outputFrames = 0;
  private lastHitMs = -Infinity;
  private running = false;

  async load(bytes: ArrayBuffer, metadata: ModelMetadata, runtimeUrl: string) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = {
      mjs: new URL("ort-wasm-simd-threaded.js", runtimeUrl).href,
      wasm: new URL("ort-wasm-simd-threaded.wasm", runtimeUrl).href,
    };
    this.metadata = metadata;
    this.session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    for (const name of ["feats", "cache0", "cache1", "cache2", "cache3"]) {
      if (!this.session.inputNames.includes(name))
        throw new Error("模型输入不符合 FSMN 流式协议。");
    }
    for (const name of [
      "probs",
      "new_cache0",
      "new_cache1",
      "new_cache2",
      "new_cache3",
    ]) {
      if (!this.session.outputNames.includes(name))
        throw new Error("模型输出不符合 FSMN 流式协议。");
    }
    // Exercise the WASM session now, so runtime failures happen before requesting a microphone.
    this.start(16000, 0.5);
    await this.audio(new Float32Array(19200));
    this.stop();
  }

  start(sampleRate: number, threshold: number) {
    if (!this.session) throw new Error("请先加载模型。");
    this.stop();
    this.frontend = new Frontend(this.metadata.cmvn);
    this.resampler = new Resampler(sampleRate);
    this.decoder = new KeywordDecoder(
      this.metadata.keyword_token_ids,
      threshold,
    );
    this.samples = 0;
    this.outputFrames = 0;
    this.lastHitMs = -Infinity;
    this.features = [];
    for (let i = 0; i < 4; i++)
      this.states[`cache${i}`] = new ort.Tensor(
        "float32",
        new Float32Array(128 * 11),
        [1, 128, 11, 1],
      );
    this.running = true;
  }

  setThreshold(value: number) {
    if (!Number.isFinite(value) || value <= 0 || value > 1)
      throw new Error("阈值必须在 0 到 1 之间。");
    if (this.decoder) this.decoder.threshold = value;
  }

  async audio(samples: Float32Array, flush = false): Promise<AudioResult> {
    if (!this.running || !this.session) throw new Error("音频流已停止。");
    const pcm = this.resampler.push(samples);
    if (!pcm.every(Number.isFinite)) throw new Error("音频包含无效采样值。");
    this.samples += pcm.length;
    this.features.push(...this.frontend.push(pcm));
    const detections: Wakeup[] = [];
    let inferenceMs = 0;
    while (this.features.length >= 3 || (flush && this.features.length)) {
      const count = Math.min(3, this.features.length);
      const features = new Float32Array(count * 400);
      for (let i = 0; i < count; i++)
        features.set(this.features.shift()!, i * 400);
      const input = new ort.Tensor("float32", features, [1, count, 400]);
      const time = performance.now();
      let outputs: ort.InferenceSession.ReturnType;
      try {
        outputs = await this.session.run({ feats: input, ...this.states });
      } finally {
        input.dispose();
      }
      inferenceMs += performance.now() - time;
      for (const state of Object.values(this.states)) state.dispose();
      for (let i = 0; i < 4; i++)
        this.states[`cache${i}`] = outputs[`new_cache${i}`];
      try {
        const probabilities = outputs.probs.data as Float32Array;
        for (let i = 0; i < count; i++) {
          const hit = this.decoder.step(
            probabilities.subarray(i * 2599, (i + 1) * 2599),
          );
          this.outputFrames++;
          if (hit) {
            this.decoder.reset();
            const startMs = Math.max(0, hit.startFrame - 8) * 30;
            const endMs = Math.max(0, hit.endFrame - 8 + 1) * 30;
            if (
              this.outputFrames >= this.metadata.model.warmup_frames &&
              endMs - this.lastHitMs >= 800
            ) {
              detections.push({
                keyword: this.metadata.keyword,
                score: hit.score,
                startMs,
                endMs,
              });
              this.lastHitMs = endMs;
            }
          }
        }
      } finally {
        outputs.probs.dispose();
      }
    }
    return {
      detections,
      audioSeconds: this.samples / 16000,
      warming: this.outputFrames < 35,
      inferenceMs,
    };
  }

  async finish() {
    // Enough trailing silence to drain right context and the resampler FIR.
    return this.audio(
      new Float32Array(Math.round(this.resampler.inputRate * 0.5)),
      true,
    );
  }

  stop() {
    this.running = false;
    this.features = [];
    for (const state of Object.values(this.states)) state.dispose();
    this.states = {};
  }
}
