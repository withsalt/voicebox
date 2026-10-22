import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";

const target = new URL("../tests/fixtures/xiaoyun.wav", import.meta.url);
const expected =
  "606ac56a00c923b080a3f6142a383f2e7691a77906646b2cc0d2e0fbbeae1e16";
const hash = (data) => createHash("sha256").update(data).digest("hex");
let existing;
try {
  existing = await readFile(target);
} catch {
  /* Download the public upstream example on demand. */
}
if (!existing || hash(existing) !== expected) {
  const response = await fetch(
    "https://www.modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun/resolve/8f8660617970f62004e6df96508ff031306a8582/example/kws_xiaoyunxiaoyun.wav",
    { signal: AbortSignal.timeout(60_000) },
  );
  if (!response.ok)
    throw new Error(`Example download failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (hash(bytes) !== expected)
    throw new Error("Example audio checksum mismatch");
  await mkdir(new URL("./", target), { recursive: true });
  await writeFile(target, bytes);
}
console.log("Upstream example audio verified (144,968 bytes).");
