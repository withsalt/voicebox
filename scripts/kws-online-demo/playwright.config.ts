import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  timeout: 120_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4179",
    channel: "chrome",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "node node_modules/vite/bin/vite.js preview --host 127.0.0.1",
    url: "http://127.0.0.1:4179",
    reuseExistingServer: !process.env.CI,
  },
});
