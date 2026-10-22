<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import Icon from "./components/Icon.vue";
import {
  models,
  cachedModelIds,
  clearModelCache,
  loadModel,
  type LoadProgress,
  type Model,
} from "./lib/cache";
import { WorkerClient } from "./lib/worker-client";
import { Microphone } from "./lib/microphone";
import type { AudioResult, Wakeup } from "./worker/engine";

type Phase =
  | "idle"
  | "loading"
  | "initializing"
  | "ready"
  | "requesting"
  | "warming"
  | "listening"
  | "file";
const phase = ref<Phase>("idle");
const selected = ref<Model | null>(
  models.find((model) => model.id === "xiaomu") ?? null,
);
const loaded = ref(false),
  cached = ref<string[]>([]),
  offlineShell = ref(false),
  online = ref(navigator.onLine);
const error = ref(""),
  notice = ref(""),
  loadingSource = ref("");
const progress = ref<LoadProgress>({
  received: 0,
  total: 0,
  stage: "downloading",
});
const threshold = ref(0.25),
  seconds = ref(0),
  inputRate = ref(0),
  latestMs = ref(0);
const levels = ref<number[]>(Array(32).fill(0));
const activeHit = ref<Wakeup | null>(null),
  hits = ref<(Wakeup & { id: number; time: string })[]>([]);
const totalHits = ref(0),
  fileProgress = ref(0),
  fileName = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
let engine: WorkerClient | null = null,
  microphone: Microphone | null = null,
  abort: AbortController | null = null;
let loadEpoch = 0,
  runEpoch = 0,
  hitTimer: ReturnType<typeof setTimeout> | undefined;
let disposed = false;
const loading = computed(
  () => phase.value === "loading" || phase.value === "initializing",
);
const active = computed(() =>
  ["requesting", "warming", "listening", "file"].includes(phase.value),
);
const percent = computed(() =>
  progress.value.total
    ? Math.min(100, (progress.value.received / progress.value.total) * 100)
    : 0,
);
const duration = computed(
  () =>
    `${Math.floor(seconds.value / 60)
      .toString()
      .padStart(2, "0")}:${Math.floor(seconds.value % 60)
      .toString()
      .padStart(2, "0")}`,
);
const status = computed(
  () =>
    ({
      idle: "等待选择模型",
      loading: "加载模型中",
      initializing: "初始化引擎",
      ready: "模型已就绪",
      requesting: "等待麦克风授权",
      warming: "音频预热中",
      listening: "正在聆听",
      file: "正在分析音频",
    })[phase.value],
);
const stageTitle = computed(() =>
  activeHit.value
    ? "已唤醒"
    : phase.value === "warming"
      ? "预热约 1 秒，再说出唤醒词"
      : phase.value === "listening"
        ? "正在聆听，试着说出"
        : phase.value === "file"
          ? "正在分析本地音频"
          : selected.value
            ? "对着麦克风，说出"
            : "选择模型，开启语音体验",
);

function describeError(value: unknown) {
  if (value instanceof DOMException) {
    if (value.name === "NotAllowedError")
      return "麦克风权限被拒绝。请在地址栏的网站权限中允许麦克风，然后重试。";
    if (value.name === "NotFoundError")
      return "没有找到麦克风，请连接输入设备后重试。";
    if (value.name === "NotReadableError")
      return "麦克风无法使用，可能被其他应用占用。请检查设备后重试。";
    if (value.name === "TimeoutError") return "下载超时，请检查网络后重试。";
    if (value.name === "EncodingError" || value.name === "NotSupportedError")
      return "无法解码此音频，请选择有效的 PCM WAV 文件。";
  }
  if (value instanceof TypeError && /fetch|network/i.test(value.message))
    return "无法下载模型。请检查网络或 ModelScope 的可用性后重试。";
  return (value instanceof Error ? value.message : String(value)).slice(0, 350);
}

async function updateCacheList() {
  const ids = await cachedModelIds();
  if (!disposed) cached.value = ids;
}

async function stopExperience(message = "") {
  runEpoch++;
  const mic = microphone;
  microphone = null;
  if (mic) await mic.stop();
  if (engine && loaded.value) void engine.call("stop").catch(() => {});
  if (loaded.value) phase.value = "ready";
  levels.value = Array(32).fill(0);
  if (message) notice.value = message;
}

async function chooseModel(model: Model) {
  const ticket = ++loadEpoch;
  abort?.abort();
  await stopExperience();
  if (ticket !== loadEpoch || disposed) return;
  engine?.terminate();
  engine = null;
  loaded.value = false;
  selected.value = model;
  phase.value = "loading";
  error.value = "";
  notice.value = "";
  activeHit.value = null;
  loadingSource.value = "";
  seconds.value = 0;
  inputRate.value = 0;
  latestMs.value = 0;
  progress.value = {
    received: 0,
    total: model.files.reduce((n, f) => n + f.size, 0),
    stage: "reading",
  };
  const controller = new AbortController();
  abort = controller;
  try {
    if (!isSecureContext)
      throw new Error(
        "请使用 HTTPS 或 localhost 打开页面，以启用安全下载、模型缓存和录音。",
      );
    const assets = await loadModel(model, controller.signal, (value) => {
      if (ticket === loadEpoch) progress.value = value;
    });
    if (ticket !== loadEpoch || disposed) return;
    loadingSource.value = assets.fromCache
      ? "从本地缓存加载"
      : "已从 ModelScope 下载";
    phase.value = "initializing";
    const client = new WorkerClient();
    engine = client;
    await client.call(
      "load",
      {
        bytes: assets.bytes,
        metadata: assets.metadata,
        runtimeUrl: new URL("ort/", document.baseURI).href,
      },
      [assets.bytes],
    );
    if (ticket !== loadEpoch || disposed) {
      client.terminate();
      return;
    }
    loaded.value = true;
    phase.value = "ready";
    if (!assets.saved)
      notice.value =
        "模型本次可用，但浏览器未能保存缓存。下次访问可能需要重新下载。";
    await updateCacheList();
    if (assets.saved) void navigator.storage?.persist?.().catch(() => false);
  } catch (reason) {
    if (ticket !== loadEpoch || disposed) return;
    engine?.terminate();
    engine = null;
    loaded.value = false;
    phase.value = "idle";
    if (!controller.signal.aborted) error.value = describeError(reason);
    await updateCacheList();
  }
}

function cancelLoad() {
  loadEpoch++;
  abort?.abort();
  abort = null;
  engine?.terminate();
  engine = null;
  loaded.value = false;
  phase.value = "idle";
  notice.value = "加载已取消，点击模型即可重新加载。";
  void updateCacheList();
}

function addResult(result: AudioResult) {
  seconds.value = result.audioSeconds;
  if (result.inferenceMs > 0) latestMs.value = result.inferenceMs;
  if (phase.value !== "file")
    phase.value = result.warming ? "warming" : "listening";
  for (const hit of result.detections) {
    totalHits.value++;
    hits.value.unshift({
      ...hit,
      id: totalHits.value,
      time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    });
    hits.value = hits.value.slice(0, 10);
    activeHit.value = hit;
    clearTimeout(hitTimer);
    hitTimer = setTimeout(() => {
      activeHit.value = null;
    }, 2200);
  }
}

async function startMicrophone() {
  if (!loaded.value || !engine || active.value) return;
  const ticket = ++runEpoch,
    client = engine,
    mic = new Microphone();
  microphone = mic;
  error.value = "";
  notice.value = "";
  activeHit.value = null;
  seconds.value = 0;
  phase.value = "requesting";
  let queuedSamples = 0,
    lastLevel = 0;
  try {
    const info = await mic.start((samples) => {
      if (ticket !== runEpoch) return;
      const length = samples.length;
      queuedSamples += length;
      if (queuedSamples > inputRate.value * 2) {
        void stopExperience();
        error.value =
          "音频处理未能跟上录音速度，已停止。请关闭占用资源的程序后重试。";
        return;
      }
      if (performance.now() - lastLevel > 65) {
        const rms = Math.sqrt(
          samples.reduce((sum, n) => sum + n * n, 0) / length,
        );
        levels.value = [...levels.value.slice(1), Math.min(1, rms * 12)];
        lastLevel = performance.now();
      }
      void client
        .call<AudioResult>("audio", { samples }, [samples.buffer])
        .then((result) => {
          if (ticket === runEpoch) addResult(result);
        })
        .catch((reason) => {
          if (ticket === runEpoch) {
            void stopExperience();
            error.value = describeError(reason);
          }
        })
        .finally(() => {
          queuedSamples -= length;
        });
    });
    if (ticket !== runEpoch) {
      await mic.stop();
      return;
    }
    inputRate.value = info.sampleRate;
    await client.call("start", {
      sampleRate: info.sampleRate,
      threshold: threshold.value,
    });
    if (ticket !== runEpoch) {
      await mic.stop();
      return;
    }
    phase.value = "warming";
    mic.connect(() => {
      if (ticket === runEpoch)
        void stopExperience("音频输入已中断，点击开始体验可重新录音。");
    });
  } catch (reason) {
    await mic.stop();
    if (ticket === runEpoch) {
      microphone = null;
      phase.value = "ready";
      error.value = describeError(reason);
    }
  }
}

async function analyzeFile(event: Event) {
  const input = event.target as HTMLInputElement,
    file = input.files?.[0];
  input.value = "";
  if (!file || !loaded.value || !engine || active.value) return;
  if (file.size > 25 * 1024 * 1024) {
    error.value = "请选择 25 MB 以内的 WAV 文件。";
    return;
  }
  const ticket = ++runEpoch,
    client = engine;
  phase.value = "file";
  fileProgress.value = 0;
  fileName.value = file.name;
  error.value = "";
  notice.value = "";
  activeHit.value = null;
  seconds.value = 0;
  let context: AudioContext | null = null;
  try {
    context = new AudioContext();
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    await context.close();
    context = null;
    if (ticket !== runEpoch) return;
    if (decoded.duration > 300) throw new Error("请选择 5 分钟以内的音频。");
    inputRate.value = decoded.sampleRate;
    const samples = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < samples.length; i++)
        samples[i] += data[i] / decoded.numberOfChannels;
    }
    await client.call("start", {
      sampleRate: decoded.sampleRate,
      threshold: threshold.value,
    });
    const size = Math.round(decoded.sampleRate * 0.08);
    for (let offset = 0; offset < samples.length; offset += size) {
      if (ticket !== runEpoch) return;
      const chunk = samples.slice(offset, offset + size);
      const result = await client.call<AudioResult>(
        "audio",
        { samples: chunk },
        [chunk.buffer],
      );
      if (ticket !== runEpoch) return;
      addResult(result);
      fileProgress.value = Math.min(1, (offset + size) / samples.length);
    }
    const tail = await client.call<AudioResult>("finish");
    if (ticket !== runEpoch) return;
    addResult(tail);
    seconds.value = decoded.duration;
    await client.call("stop");
    phase.value = "ready";
    notice.value = `已完成「${file.name}」的本地分析，未上传音频。`;
  } catch (reason) {
    if (ticket === runEpoch) {
      await stopExperience();
      error.value = describeError(reason);
    }
  } finally {
    if (context && context.state !== "closed")
      await context.close().catch(() => {});
  }
}

function updateThreshold(event: Event) {
  threshold.value = Number((event.target as HTMLInputElement).value);
  const ticket = runEpoch;
  if (engine && loaded.value)
    void engine
      .call("threshold", { value: threshold.value })
      .catch((reason) => {
        if (ticket === runEpoch) {
          void stopExperience();
          error.value = describeError(reason);
        }
      });
}

async function removeCache() {
  try {
    await clearModelCache();
    await updateCacheList();
    notice.value = "模型缓存已清除。当前已加载的模型仍可继续使用。";
  } catch {
    error.value = "无法清除缓存，请检查浏览器的存储权限。";
  }
}

function modelKey(event: KeyboardEvent, index: number) {
  const shift =
    event.key === "ArrowDown" || event.key === "ArrowRight"
      ? 1
      : event.key === "ArrowUp" || event.key === "ArrowLeft"
        ? -1
        : 0;
  if (!shift && event.key !== "Home" && event.key !== "End") return;
  event.preventDefault();
  const next =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? models.length - 1
        : (index + shift + models.length) % models.length;
  (
    document.querySelectorAll(".model-option")[next] as HTMLButtonElement
  ).focus();
  void chooseModel(models[next]);
}

const updateOnline = () => {
  online.value = navigator.onLine;
};
const visibility = () => {
  if (document.hidden && active.value)
    void stopExperience("页面进入后台，体验已停止。返回后可重新开始。");
};
const pageExit = () => {
  loadEpoch++;
  runEpoch++;
  abort?.abort();
  void microphone?.stop();
  microphone = null;
  engine?.terminate();
  engine = null;
  loaded.value = false;
  phase.value = "idle";
  levels.value = Array(32).fill(0);
};
onMounted(() => {
  void updateCacheList();
  if (selected.value) void chooseModel(selected.value);
  addEventListener("online", updateOnline);
  addEventListener("offline", updateOnline);
  document.addEventListener("visibilitychange", visibility);
  addEventListener("pagehide", pageExit);
  if (import.meta.env.PROD && isSecureContext && "serviceWorker" in navigator) {
    void navigator.serviceWorker
      .register(new URL("sw.js", document.baseURI).href)
      .then(() => navigator.serviceWorker.ready)
      .then(() => {
        offlineShell.value = true;
      })
      .catch(() => {});
  }
});
onBeforeUnmount(() => {
  disposed = true;
  loadEpoch++;
  runEpoch++;
  abort?.abort();
  pageExit();
  clearTimeout(hitTimer);
  removeEventListener("online", updateOnline);
  removeEventListener("offline", updateOnline);
  document.removeEventListener("visibilitychange", visibility);
  removeEventListener("pagehide", pageExit);
});
</script>

<template>
  <div class="site-wrap">
    <header class="site-header">
      <a
        class="brand"
        href="https://github.com/withsalt/voicebox"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="VoiceBox GitHub 仓库"
        ><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span
        >VoiceBox<span class="brand-divider"></span
        ><span class="brand-section">语音实验室</span></a
      >
      <nav class="project-links" aria-label="项目资源">
        <a
          class="source-link"
          href="https://github.com/withsalt/voicebox-kws-train-opensource"
          target="_blank"
          rel="noopener noreferrer"
          >训练仓库 <Icon name="arrow"
        /></a>
        <a
          class="source-link"
          href="https://modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun"
          target="_blank"
          rel="noopener noreferrer"
          >原始项目 <Icon name="arrow"
        /></a>
        <a
          class="source-link"
          href="https://www.modelscope.cn/models/withsalt/voicebox_models/tree/master/v1/kws"
          target="_blank"
          rel="noopener noreferrer"
          >唤醒词模型下载 <Icon name="arrow"
        /></a>
      </nav>
    </header>
    <main>
      <section class="intro">
        <h1>唤醒词模型在线体验</h1>
        <p>
          模型按需加载，当前体验的是 ONNX 模型 Web 推理，支持 Python、C# 等语言的推理后端。
        </p>
        <div class="feature-tags">
          <span><Icon name="lock" /> 音频本地处理</span
          ><span><Icon name="download" /> 模型自动缓存</span
          ><span><Icon name="headphones" /> 即开即用</span>
        </div>
      </section>
      <div v-if="error" class="message error-message" role="alert">
        <p>{{ error }}</p>
        <button
          class="icon-button"
          aria-label="关闭错误提示"
          @click="error = ''"
        >
          <Icon name="close" />
        </button>
      </div>
      <div v-if="notice" class="message notice-message" role="status">
        <p>{{ notice }}</p>
        <button class="icon-button" aria-label="关闭提示" @click="notice = ''">
          <Icon name="close" />
        </button>
      </div>
      <div v-if="!online" class="offline-note">
        当前网络离线，已缓存的模型仍可使用。
      </div>
      <div class="workspace">
        <section class="model-panel" aria-labelledby="model-title">
          <div class="panel-heading">
            <h2 id="model-title"><span class="step">01</span> 选择唤醒模型</h2>
            <span class="count-label">4 个模型</span>
          </div>
          <p class="panel-description">一种呼唤，一个新的开始。</p>
          <div class="model-list" role="radiogroup" aria-label="唤醒模型">
            <button
              v-for="(model, index) in models"
              :key="model.id"
              class="model-option"
              role="radio"
              :aria-checked="selected?.id === model.id"
              :tabindex="
                selected
                  ? selected.id === model.id
                    ? 0
                    : -1
                  : index === 0
                    ? 0
                    : -1
              "
              @click="chooseModel(model)"
              @keydown="modelKey($event, index)"
            >
              <span class="model-avatar" aria-hidden="true">{{
                model.label
              }}</span>
              <span class="model-copy"
                ><span class="model-name">{{ model.keyword }}</span
                ><span class="model-meta"
                  >FSMN <span>·</span>
                  {{
                    (model.files.reduce((n, f) => n + f.size, 0) / 1e6).toFixed(
                      2,
                    )
                  }}
                  MB</span
                ><span v-if="model.id === 'xiaoyun'" class="model-description"
                  >小云小云由官方开源仓库训练</span
                ></span
              >
              <span class="model-side"
                ><span v-if="cached.includes(model.id)" class="cached-label"
                  >已缓存</span
                ><span class="radio-mark" aria-hidden="true"
                  ><Icon v-if="selected?.id === model.id" name="check" /></span
              ></span>
            </button>
          </div>
          <div class="model-footnote">
            <Icon name="download" />
            <p>默认加载小木小木，首次使用自动下载<br />再次使用，优先读取本地缓存</p>
          </div>
        </section>

        <section
          class="experience-panel"
          :class="{
            'is-hit': activeHit,
            'is-listening': phase === 'listening',
          }"
          aria-labelledby="experience-title"
        >
          <div class="panel-heading">
            <h2 id="experience-title">
              <span class="step">02</span> 试着呼唤它
            </h2>
            <span
              class="status-pill"
              :class="{ live: phase === 'listening', ready: loaded && !active }"
              role="status"
              ><i></i>{{ status }}</span
            >
          </div>
          <div class="experience-stage">
            <p class="stage-prompt" aria-live="polite">{{ stageTitle }}</p>
            <h3 class="keyword" :class="{ placeholder: !selected }">
              {{
                activeHit?.keyword || selected?.keyword || "每一声，都有回应"
              }}
            </h3>
            <div
              class="mic-orbit"
              :class="{ listening: active, success: activeHit }"
              aria-hidden="true"
            >
              <div class="mic-disc">
                <Icon :name="activeHit ? 'check' : 'mic'" />
              </div>
            </div>
            <template v-if="loading">
              <div class="load-area">
                <div class="progress-label">
                  <span>{{
                    phase === "initializing"
                      ? "准备本地推理引擎…"
                      : progress.stage === "reading"
                        ? "读取本地缓存…"
                        : progress.stage === "verifying"
                          ? "校验模型完整性…"
                          : "从 ModelScope 下载模型…"
                  }}</span
                  ><span>{{ Math.round(percent) }}%</span>
                </div>
                <progress
                  :value="percent"
                  max="100"
                  aria-label="模型下载进度"
                ></progress>
                <div class="load-details">
                  <span
                    >{{ (progress.received / 1e6).toFixed(2) }} /
                    {{ (progress.total / 1e6).toFixed(2) }} MB</span
                  ><button class="text-button" @click="cancelLoad">
                    取消加载
                  </button>
                </div>
              </div>
              <p class="runtime-note">
                首次使用还需加载共用运行时，后续自动复用。
              </p>
            </template>
            <template v-else>
              <div
                class="waveform"
                :class="{ muted: !active }"
                role="img"
                :aria-label="active ? '麦克风音量' : '等待音频输入'"
              >
                <span
                  v-for="(level, i) in levels"
                  :key="i"
                  :style="{ height: `${3 + level * 28}px` }"
                ></span>
              </div>
              <p class="wave-caption">
                {{
                  phase === "file"
                    ? `${fileName} · ${Math.round(fileProgress * 100)}%`
                    : active
                      ? `${duration} · ${phase === "requesting" ? "请允许麦克风权限" : phase === "warming" ? "正在预热音频" : "正在聆听你的声音"}`
                      : loaded
                        ? `${loadingSource} · 准备就绪`
                        : selected
                          ? "点击左侧模型可重新加载"
                          : "从左侧选择一个模型开始"
                }}
              </p>
              <button
                class="record-button"
                :class="{ recording: active }"
                :disabled="!loaded"
                @click="active ? stopExperience() : startMicrophone()"
              >
                <Icon :name="active ? 'stop' : 'mic'" />{{
                  active ? "停止体验" : "开始体验"
                }}
              </button>
              <p class="privacy-caption">
                <Icon name="lock" />音频仅在设备上处理，不会上传
              </p>
            </template>
          </div>
          <div class="threshold-setting">
            <div class="threshold-heading">
              <label for="threshold">唤醒阈值</label>
              <output for="threshold">{{ threshold.toFixed(2) }}</output>
            </div>
            <input
              id="threshold"
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              :value="threshold"
              aria-describedby="threshold-help"
              @input="updateThreshold"
            />
            <p id="threshold-help">
              越低越容易唤醒，越高越严格。默认 0.25，调整实时生效。
            </p>
          </div>
          <div class="experience-bottom">
            <span
              ><i></i>
              {{
                activeHit
                  ? `检测得分 ${(activeHit.score * 100).toFixed(1)}%`
                  : "浏览器本地推理"
              }}</span
            ><button
              class="text-button upload-button"
              :disabled="!loaded || active || loading"
              @click="fileInput?.click()"
            >
              <Icon name="file" />测试本地 WAV</button
            ><input
              ref="fileInput"
              type="file"
              accept=".wav,audio/wav,audio/x-wav"
              class="sr-only"
              aria-label="选择本地 WAV 文件"
              @change="analyzeFile"
            />
          </div>
        </section>
      </div>

      <section class="history-panel" aria-labelledby="history-title">
        <div class="panel-heading">
          <h2 id="history-title">
            唤醒记录 <span class="history-count">{{ totalHits }}</span>
          </h2>
          <button
            class="text-button"
            :disabled="!hits.length"
            @click="
              hits = [];
              totalHits = 0;
            "
          >
            清空记录
          </button>
        </div>
        <p v-if="!hits.length" class="empty-history">
          {{
            loaded
              ? "准备好了。试着说出唤醒词，第一条记录会出现在这里。"
              : "你的每一次呼唤，都将在这里留下回应。"
          }}
        </p>
        <ol v-else class="history-list" aria-live="polite">
          <li v-for="hit in hits" :key="hit.id">
            <span class="hit-check"><Icon name="check" /></span
            ><strong>{{ hit.keyword }}</strong
            ><span class="hit-score"
              >得分 {{ (hit.score * 100).toFixed(1) }}%</span
            ><span class="hit-position"
              >音频 {{ (hit.startMs / 1000).toFixed(2) }}s</span
            ><time>{{ hit.time }}</time>
          </li>
        </ol>
        <p v-if="hits.length" class="score-note">
          显示最近 10 条；检测得分用于判断唤醒，不代表识别准确率。
        </p>
      </section>

      <details class="advanced-panel">
        <summary>高级设置 <Icon name="chevron" /></summary>
        <div class="advanced-content">
          <div class="cache-setting">
            <div>
              <span>模型本地缓存</span
              ><small>已缓存 {{ cached.length }} / 4 个模型</small>
            </div>
            <button
              class="secondary-button"
              :disabled="loading || active || !cached.length"
              @click="removeCache"
            >
              <Icon name="trash" />清除模型缓存
            </button>
          </div>
          <p class="engine-info">
            WASM · 单线程<span v-if="inputRate">
              · 输入 {{ (inputRate / 1000).toFixed(1) }} kHz → 16 kHz</span
            ><span v-if="latestMs">
              · 最近推理 {{ latestMs.toFixed(1) }} ms</span
            >
          </p>
          <p class="storage-note">
            缓存可能被浏览器或系统清理。录音需要 HTTPS 或
            localhost；页面进入后台会停止体验。
          </p>
        </div>
      </details>
    </main>
    <footer>
      <span>VOICEBOX <span class="footer-divider">/</span> ON-DEVICE AUDIO</span
      ><span :class="{ 'offline-ready': offlineShell }">{{
        offlineShell
          ? "页面资源已缓存，可离线打开"
          : "轻量模型 · 本地推理 · 保护隐私"
      }}</span>
    </footer>
  </div>
</template>
