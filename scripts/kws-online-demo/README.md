# VoiceBox · 浏览器语音唤醒体验

独立的静态网页应用：Vue 3 + TypeScript + Vite + ONNX Runtime Web。支持四组 FSMN KWS 模型、按需下载和校验、浏览器本地缓存、麦克风唤醒及本地 WAV 测试。音频不上传，无需推理后端。

## 启动

需要 Node.js **22.12 或更新版本**与 pnpm。首次安装需要联网下载依赖。

```powershell
cd E:\Projects\voicebox\scripts\kws-online-demo
pnpm install
pnpm dev
```

打开 <http://127.0.0.1:5173>，默认选中并加载“小木小木”，也可切换其他模型。等待“模型已就绪”，再点击“开始体验”并允许麦克风权限。开始后有约一秒预热，请等到“正在聆听”后说出唤醒词。也可点击“测试本地 WAV”，选择 25 MB、5 分钟以内的音频。

模型选择如下，文字直接来自模型的元数据：

| 模型                | 唤醒词   | 模型与配置合计 |
| ------------------- | -------- | -------------- |
| `kws-xiaoai-onnx`   | 小爱同学 | 3.13 MB        |
| `kws-xiaomu-onnx`   | 小木小木 | 3.06 MB        |
| `kws-xiaowang-onnx` | 小王同学 | 3.13 MB        |
| `kws-xiaoyun-onnx`  | 小云小云 | 3.13 MB        |

小云小云由官方开源仓库训练。页面提供[训练仓库](https://github.com/withsalt/voicebox-kws-train-opensource)、[原始项目](https://modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun)与[唤醒词模型下载](https://www.modelscope.cn/models/withsalt/voicebox_models/tree/master/v1/kws)链接。

首次初始化还需共用的 WASM 运行时，未压缩约 14.24 MB。安装脚本从锁定的 `onnxruntime-web@1.30.0` 依赖复制运行时到 `public/ort/`，无需手动下载；模块入口以 `.js` 扩展名发布，并显式配置模块与 WASM 路径，以兼容 EdgeOne 等静态托管服务。模型从 [ModelScope](https://www.modelscope.cn/models/withsalt/voicebox_models/tree/master/v1/kws) 获取，未把权重打包进代码或发布目录。

## 构建与离线使用

```powershell
pnpm build
pnpm preview
```

打开 <http://127.0.0.1:4179>。部署时将 `dist/` 作为完整静态目录发布，支持根路径或子目录；必须使用 **HTTPS**，开发可用 localhost/回环地址。不要直接以 `file://` 打开 HTML。普通 HTTP 局域网 IP 通常不能录音。

生产构建生成 Service Worker，首次访问会缓存页面、Worker、AudioWorklet 和 WASM。页脚出现“页面资源已缓存，可离线打开”后，可断网重新打开。只有事先下载过且仍在浏览器缓存中的模型能够离线加载。开发模式不注册 Service Worker，调试离线请用生产预览。

建议静态服务器开启 Brotli/gzip，正确返回 `.js` 的 JavaScript MIME（`application/javascript` 或 `text/javascript`）和 `.wasm` 的 `application/wasm`。为 `sw.js` 设置 `Cache-Control: no-cache`，保留脚本内置的更新检查。页面资源缓存和模型缓存使用独立命名空间；高级设置中的“清除模型缓存”不删除站点离线资源。

### EdgeOne 部署排错

如果出现 `Failed to fetch dynamically imported module: .../ort/ort-wasm-simd-threaded.mjs`，且该请求虽然返回 200，但 `Content-Type` 为 `application/octet-stream`，浏览器会拒绝执行该模块。这是模块入口的 MIME 类型错误，不是 WASM 不支持或模型损坏。

当前构建已使用 `ort/ort-wasm-simd-threaded.js` 规避这一问题。重新运行 `pnpm build`，然后将完整 `dist/` 重新部署到 EdgeOne，不能只改服务器上的文件名。部署后刷新页面；若仍显示旧 `.mjs` 请求，清理 CDN 对 `index.html` 和 `sw.js` 的缓存，再在浏览器更新或注销旧 Service Worker 后重试。旧版也可通过将 `.mjs` 响应的 `Content-Type` 设置为 `application/javascript` 临时修复。

若已有旧版 Service Worker 影响开发，可在浏览器开发者工具 Application → Service Workers 注销当前站点的 worker，再刷新。

## 测试

```powershell
pnpm test
pnpm build
pnpm test:e2e
```

- 单元测试不需要模型或网络，覆盖 Fbank/LFR/CMVN 数值参考、分块一致性、44.1/48 kHz 重采样与抗混叠、CTC 重复字/过期状态、文件完整性。
- 浏览器测试需要**已安装的 Google Chrome**以及连接 ModelScope 的网络。自动下载一份约 145 KB 的上游“小云小云” WAV 样例，校验 SHA-256 后保存到被 Git 忽略的 `tests/fixtures/`。
- 浏览器测试覆盖四模型下载、损坏缓存重下、完整离线刷新、本地 WAV 检出、模拟麦克风检出、停止/切换时释放轨道、权限拒绝、加载取消、缓存写入失败和手机布局。
- 模拟麦克风使用 Chrome 的测试输入，实际用户的麦克风、环境噪声、距离和误唤醒率仍需真机验证。
- Fbank 数值参考由 NumPy 的独立 FFT/滤波矩阵生成；需要更新时运行 `python scripts/generate-golden.py`（可选，需安装 NumPy）。

## 代码结构

```text
src/App.vue                 页面、下载与录音状态、结果记录
src/models.json             固定版本模型清单、大小和 SHA-256
src/lib/cache.ts            校验、进度、取消、本地缓存与配额回退
src/lib/microphone.ts       麦克风权限、AudioContext 与资源释放
src/dsp/resampler.ts        保留跨块状态的低通 sinc 重采样
src/dsp/frontend.ts         Kaldi Fbank → LFR(5,3) → CMVN
src/dsp/decoder.ts          帧同步 CTC prefix beam search
src/worker/engine.ts        WASM 会话、声学 cache、预热、防抖
src/worker/kws.worker.ts    串行推理消息循环
public/audio-capture.js     20 ms 单声道 PCM 采集
scripts/build-sw.mjs        构建后生成离线资源缓存
tests/                     算法与浏览器测试
```

## 行为与限制

- 首页默认选中并加载“小木小木”，首次使用下载 ONNX 和 `kws_meta.json`，再次使用优先读取本地缓存；点击“开始体验”后才请求麦克风。四组模型版本固定为 `8a8cb13c7f299344409dbc32b4dd0e327c534133`。更新时同步修改清单中的 URL、大小、SHA-256 和配置校验。
- 使用单线程 WASM，无需 WebGPU、COOP/COEP 或推理服务器。特征提取与推理均在独立 Web Worker 运行。
- 不假定设备提供 16 kHz；读取 AudioContext 的实际采样率，在 Worker 中低通重采样到 16 kHz。
- 默认请求回声消除，关闭浏览器降噪和自动增益；设备是否采纳由浏览器决定。
- 90 ms 特征一批；模型固有约 240 ms lookahead，另有前端和采集调度延迟。“最近推理”只计算模型调用，不是端到端唤醒延迟。
- 体验区可拖动滑块实时调整唤醒阈值，范围 0.05–0.95，步长 0.05，默认 0.25。阈值越低越容易触发，也可能增加误唤醒。防抖 800 ms。得分采用现有实现的 `sqrt(各 token 峰值后验乘积)`，不是识别准确率。
- 只保留当前模型会话，切换模型会停止录音和重置流。后台/锁屏或输入中断时停止体验，返回后需重新开始。
- PCM 队列超过两秒时停止并提示，不静默丢帧；录音停止后立即关闭音频轨道。
- 缓存可能因配额、隐私模式或系统清理而失效；缓存写入失败时仍可使用本次内存中的模型。
- 首轮验证面向桌面 Chrome；Edge、Android Chrome、Safari/iOS 和 Firefox 的真实设备效果需另行验证。

## 许可与来源

音频前端和 CTC 解码根据本仓库 `VoiceBox.Wakeup.FsmnKws` 实现移植，遵循仓库根目录的 MIT 许可证。Vue 和 ONNX Runtime 也使用 MIT 许可证；发布产物包含对应的许可说明。模型权重及测试音频来自各自 ModelScope 仓库，许可由上游决定，不因网页代码的许可证改变。

模型通过浏览器直接下载，使用 `credentials: omit`，不携带 ModelScope 登录凭据。若上游修改 CORS 策略，需要另行配置资源代理或获得许可后部署镜像。
