import { copyFile, mkdir, rm } from "node:fs/promises";

const root = new URL("../", import.meta.url);
await mkdir(new URL("public/ort/", root), { recursive: true });
// Some static hosts serve .mjs as application/octet-stream, which browsers reject
// for module imports. Keep the upstream content and use the widely supported .js suffix.
for (const [source, destination] of [
  ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.js"],
  ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.wasm"],
]) {
  await copyFile(
    new URL(`node_modules/onnxruntime-web/dist/${source}`, root),
    new URL(`public/ort/${destination}`, root),
  );
}
await rm(new URL("public/ort/ort-wasm-simd-threaded.mjs", root), { force: true });
