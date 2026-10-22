import catalog from "../models.json";
import { parseMetadata } from "./metadata";

export const models = catalog;
export type Model = (typeof models)[number];
const CACHE = "voicebox-kws-models-v1";
export interface LoadProgress {
  received: number;
  total: number;
  stage: "reading" | "downloading" | "verifying";
}

async function openCache() {
  try {
    return await caches.open(CACHE);
  } catch {
    return null;
  }
}

export async function cachedModelIds(): Promise<string[]> {
  const cache = await openCache();
  if (!cache) return [];
  const result = [];
  for (const model of models) {
    try {
      const files = await Promise.all(
        model.files.map((file) => cache.match(file.url)),
      );
      if (files.every(Boolean)) result.push(model.id);
    } catch {
      /* Storage may be unavailable in private mode. */
    }
  }
  return result;
}

export async function verifyAsset(
  bytes: ArrayBuffer,
  asset: { size: number; sha256: string },
) {
  if (bytes.byteLength !== asset.size)
    throw new Error("模型文件大小不正确，请重新下载。");
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  const hex = [...new Uint8Array(hash)]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  if (hex !== asset.sha256) throw new Error("模型完整性校验失败，请重新下载。");
}

export async function loadModel(
  model: Model,
  signal: AbortSignal,
  progress: (p: LoadProgress) => void,
) {
  const cache = await openCache();
  const total = model.files.reduce((n, f) => n + f.size, 0);
  let received = 0,
    allCached = true,
    saved = Boolean(cache);
  const assets = new Map<string, ArrayBuffer>();
  for (const file of model.files) {
    signal.throwIfAborted();
    let bytes: ArrayBuffer | undefined;
    if (cache) {
      try {
        const response = await cache.match(file.url);
        if (response) {
          const candidate = await response.arrayBuffer();
          await verifyAsset(candidate, file);
          bytes = candidate;
          progress({ received: received + file.size, total, stage: "reading" });
        }
      } catch {
        await cache.delete(file.url).catch(() => {});
      }
    }
    if (!bytes) {
      allCached = false;
      progress({ received, total, stage: "downloading" });
      const requestSignal = AbortSignal.any([
        signal,
        AbortSignal.timeout(60_000),
      ]);
      const response = await fetch(file.url, {
        signal: requestSignal,
        credentials: "omit",
      });
      if (!response.ok)
        throw new Error(
          `模型下载失败（HTTP ${response.status}），请稍后重试。`,
        );
      const reader = response.body?.getReader();
      if (!reader)
        throw new Error("浏览器无法读取下载流，请使用最新版 Chrome 或 Edge。");
      const output = new Uint8Array(file.size);
      let offset = 0;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          if (offset + value.length > output.length)
            throw new Error("下载内容超过预期大小，请重试。");
          output.set(value, offset);
          offset += value.length;
          progress({
            received: received + offset,
            total,
            stage: "downloading",
          });
        }
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
      if (offset !== file.size)
        throw new Error("下载未完成，请检查网络后重试。");
      bytes = output.buffer;
      progress({ received: received + file.size, total, stage: "verifying" });
      await verifyAsset(bytes, file);
      signal.throwIfAborted();
      if (cache) {
        try {
          await cache.put(file.url, new Response(bytes.slice(0)));
        } catch {
          saved = false;
        }
      }
    }
    assets.set(file.name, bytes);
    received += file.size;
  }
  signal.throwIfAborted();
  const metadata = parseMetadata(assets.get("kws_meta.json")!, model.keyword);
  return {
    metadata,
    bytes: assets.get("kws_streaming.onnx")!,
    fromCache: allCached,
    saved,
  };
}

export async function clearModelCache() {
  if (typeof caches !== "undefined") await caches.delete(CACHE);
}
