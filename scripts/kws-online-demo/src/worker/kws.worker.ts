import { KwsEngine } from "./engine";

const engine = new KwsEngine();
let queue = Promise.resolve();
self.onmessage = (event) => {
  const { id, type, payload } = event.data;
  queue = queue.then(async () => {
    try {
      let result: unknown;
      switch (type) {
        case "load":
          await engine.load(
            payload.bytes,
            payload.metadata,
            payload.runtimeUrl,
          );
          break;
        case "start":
          engine.start(payload.sampleRate, payload.threshold);
          break;
        case "audio":
          result = await engine.audio(payload.samples);
          break;
        case "finish":
          result = await engine.finish();
          break;
        case "threshold":
          engine.setThreshold(payload.value);
          break;
        case "stop":
          engine.stop();
          break;
        default:
          throw new Error("未知的音频指令。");
      }
      self.postMessage({ id, result });
    } catch (error) {
      engine.stop();
      self.postMessage({
        id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });
};
