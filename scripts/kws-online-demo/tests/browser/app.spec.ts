import { test, expect, chromium } from "@playwright/test";
import { resolve } from "node:path";
import models from "../../src/models.json" with { type: "json" };

test("loads the default and all four models, repairs damaged cache, and works after an offline reload", async ({
  page,
  context,
}) => {
  let downloads = 0;
  const runtimeRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/resolve/")) downloads++;
    const path = new URL(request.url()).pathname;
    if (path.includes("/ort/")) runtimeRequests.push(path);
  });
  await page.goto("/");
  await expect(
    page.getByRole("radio", { name: /小木小木/ }),
  ).toHaveAttribute("aria-checked", "true");
  await expect(page.getByRole("slider", { name: "唤醒阈值" })).toHaveValue("0.25");
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  expect(runtimeRequests).toContain("/ort/ort-wasm-simd-threaded.js");
  expect(runtimeRequests).toContain("/ort/ort-wasm-simd-threaded.wasm");
  expect(runtimeRequests.some((path) => path.endsWith(".mjs"))).toBe(false);
  expect(downloads).toBe(2);
  for (const model of models) {
    if (model.id === "xiaomu") continue;
    await page.getByRole("radio", { name: new RegExp(model.keyword) }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "模型已就绪" }),
    ).toBeVisible({ timeout: 60_000 });
  }
  expect(downloads).toBe(8);
  await page.evaluate(async (url) => {
    const cache = await caches.open("voicebox-kws-models-v1");
    await cache.put(url, new Response("corrupted"));
  }, models[0].files[1].url);
  await page.getByRole("radio", { name: /小爱同学/ }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  expect(downloads).toBe(9);
  await expect(page.getByText("页面资源已缓存，可离线打开")).toBeVisible();
  await page.evaluate(async () => {
    if (!navigator.serviceWorker.controller)
      await new Promise<void>((resolve) =>
        navigator.serviceWorker.addEventListener(
          "controllerchange",
          () => resolve(),
          { once: true },
        ),
      );
  });
  await context.setOffline(true);
  await page.reload();
  await expect(page.locator(".cached-label")).toHaveCount(4);
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible();
  await page.getByRole("radio", { name: /小云小云/ }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible();
  await expect(page.getByText("从本地缓存加载 · 准备就绪")).toBeVisible();
  expect(downloads).toBe(9);
});

test("detects the real upstream keyword in a local WAV with the full browser audio pipeline", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("radio", { name: /小云小云/ }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  await page
    .getByLabel("选择本地 WAV 文件")
    .setInputFiles(resolve("tests/fixtures/xiaoyun.wav"));
  await expect(page.locator(".history-list li")).toHaveCount(1, {
    timeout: 30_000,
  });
  await expect(page.locator(".history-list strong")).toHaveText("小云小云");
  await expect(page.locator(".hit-score")).toContainText("99.");
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible();
  await expect(page.getByText(/已完成「xiaoyun.wav」/)).toBeVisible();
});

test("remains usable after a microphone permission refusal", async ({
  page,
  context,
}) => {
  await context.clearPermissions();
  await page.goto("/");
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "开始体验", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("麦克风");
  await expect(
    page.getByRole("button", { name: "开始体验", exact: true }),
  ).toBeEnabled();
});

test("responsive layout and radio keyboard navigation", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const first = page.getByRole("radio", { name: /小爱同学/ });
  await first.focus();
  await first.press("ArrowRight");
  await expect(page.getByRole("radio", { name: /小木小木/ })).toBeFocused();
  await expect(page.getByRole("radio", { name: /小木小木/ })).toHaveAttribute(
    "aria-checked",
    "true",
  );
});

test("captures a synthetic microphone, detects speech, and releases tracks on stop and model switch", async () => {
  const browser = await chromium.launch({
    channel: "chrome",
    args: [
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-audio-capture=${resolve("tests/fixtures/xiaoyun.wav")}`,
    ],
  });
  try {
    const context = await browser.newContext({ permissions: ["microphone"] });
    const page = await context.newPage();
    await page.addInitScript(() => {
      const streams: MediaStream[] = [];
      (window as unknown as { testStreams: MediaStream[] }).testStreams =
        streams;
      const original = navigator.mediaDevices.getUserMedia.bind(
        navigator.mediaDevices,
      );
      navigator.mediaDevices.getUserMedia = async (constraints) => {
        const stream = await original(constraints);
        streams.push(stream);
        return stream;
      };
    });
    await page.goto("http://127.0.0.1:4179/");
    await page.getByRole("radio", { name: /小云小云/ }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "模型已就绪" }),
    ).toBeVisible({ timeout: 60_000 });
    await page.getByRole("button", { name: "开始体验", exact: true }).click();
    await expect(page.locator(".history-list li")).toHaveCount(1, {
      timeout: 20_000,
    });
    await page.getByRole("button", { name: "停止体验", exact: true }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "模型已就绪" }),
    ).toBeVisible();
    expect(
      await page.evaluate(() =>
        (window as unknown as { testStreams: MediaStream[] }).testStreams.every(
          (s) => s.getTracks().every((t) => t.readyState === "ended"),
        ),
      ),
    ).toBe(true);
    await page.getByRole("button", { name: "开始体验", exact: true }).click();
    await expect(
      page.getByRole("status").filter({ hasText: /音频预热中|正在聆听/ }),
    ).toBeVisible();
    await page.getByRole("radio", { name: /小爱同学/ }).click();
    await expect(
      page.getByRole("status").filter({ hasText: "模型已就绪" }),
    ).toBeVisible({ timeout: 60_000 });
    expect(
      await page.evaluate(() =>
        (window as unknown as { testStreams: MediaStream[] }).testStreams.every(
          (s) => s.getTracks().every((t) => t.readyState === "ended"),
        ),
      ),
    ).toBe(true);
  } finally {
    await browser.close();
  }
});

test("can cancel a pending download and switch models without applying stale results", async ({
  page,
}) => {
  let release!: () => void;
  const paused = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/resolve/**", async (route) => {
    await paused;
    await route.abort().catch(() => {});
  });
  await page.goto("/");
  await page.getByRole("radio", { name: /小爱同学/ }).click();
  await page.getByRole("button", { name: "取消加载", exact: true }).click();
  await expect(
    page.getByText("加载已取消，点击模型即可重新加载。"),
  ).toBeVisible();
  release();
  await page.unroute("**/resolve/**");
  await page.getByRole("radio", { name: /小木小木/ }).click();
  await page.getByRole("radio", { name: /小云小云/ }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(page.locator(".keyword")).toHaveText("小云小云");
  await expect(page.getByRole("alert")).toHaveCount(0);
});

test("keeps the model usable in memory when browser cache writes fail", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const open = caches.open.bind(caches);
    caches.open = async (name) => {
      const cache = await open(name);
      if (name === "voicebox-kws-models-v1")
        cache.put = async () => {
          throw new DOMException("Full", "QuotaExceededError");
        };
      return cache;
    };
  });
  await page.goto("/");
  await expect(
    page.getByRole("status").filter({ hasText: "模型已就绪" }),
  ).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/浏览器未能保存缓存/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "开始体验", exact: true }),
  ).toBeEnabled();
  await expect(page.locator(".cached-label")).toHaveCount(0);
});
