import { expect, it } from "vitest";
import { verifyAsset } from "../src/lib/cache";

it("rejects truncated and corrupted model assets", async () => {
  const bytes = new TextEncoder().encode("VoiceBox");
  const digest = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  ]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
  const asset = { size: bytes.length, sha256: digest };
  await expect(verifyAsset(bytes.buffer, asset)).resolves.toBeUndefined();
  await expect(verifyAsset(bytes.buffer.slice(1), asset)).rejects.toThrow(
    "大小",
  );
  bytes[0] = 0;
  await expect(verifyAsset(bytes.buffer, asset)).rejects.toThrow("校验");
});
