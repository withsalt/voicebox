# AEC Challenge 双讲打断评测

该工具使用微软 AEC Challenge 本地数据，衡量 `VoiceBox.Assistant` 当前播放期
barge-in 声学判定是否可靠。benchmark 不实例化或编排一套平行检测算法，而是创建
真实 `VoiceAssistant`，只用 `BenchmarkDuplexAudioIngress` 替换硬件双工采集服务。
每个数据集帧随后经过生产运行时中的：

1. WebRTC APM 主 AEC；
2. render-aware DTD 及 render/capture 时间线对齐；
3. 当前 barge-in 窗口累计规则；
4. `ActiveNearEndCoordinator`、turn 状态机和真实 `BargeInValidationKind.Heard` 事件。

`VoiceAssistantBenchmarkProbe` 是被动只读探针，只补充概率峰值和未触发诊断，
不参与裁决。生产 public 构造函数和默认时钟均未改变；benchmark 通过 internal friend
入口注入数据集时钟，使 startup guard、AEC 指标采样节奏、指标新鲜度和 decider
随音频样本推进。WebRTC APM 的处理和指标数值保持不变，只把指标观测时刻映射到
同一数据集时间轴。

当真实控制面已经接收低频 near-end confirmation 时，评测器会在该 callback 返回后
给单一非音频 consumer 让出调度机会，再推进下一段虚拟音频。这个等待不推进数据集
时钟、不改变 PCM/DTD 输入或裁决，并且只通过被动 probe 确认入队 confirmation 已被处理；
等待上限为当前 callback period，超过上限将作为评测错误而不是悄悄推进虚拟音频。

## 三层验收

完整的“能正常打断”建议同时满足三层：

- **声学运行时层**：本工具检查双讲召回、远端单讲误触发和近端起点到
  `Interrupt` 声学判定的延迟。检测结果只取真实 Assistant 发出的
  `BargeInValidationKind.Heard`，不会直接读取 decider 的 `Interrupt` 作为成功。
- **控制层**：`VoiceBox.Assistant.Tests` 检查 ASR 验证、ducking、取消 LLM/TTS、
  清空播放队列以及状态转移。
- **中文端到端层**：`--mode e2e` 使用真实 VoxCPM TTS/Qwen3 ASR，检查 render 捕获、
  双讲声学链路、`Heard -> Accepted/Rejected`、CER 和播放停止延迟。

只运行其中一层不能证明端到端打断正常。

## 数据映射

| AEC Challenge | VoiceBox 输入 |
|---|---|
| `*_lpb.wav` / `farend_speech` | `RenderReference` |
| `*_mic.wav` / `nearend_mic_signal` | 主 AEC 前采集 |
| WebRTC APM 输出 | `CaptureAfterAec` |
| `nearend_speech * nearend_scale` | synthetic 近端真值 |

synthetic 真值使用透明且可复现的能量规则：20 ms RMS 帧，门槛为
`max(0.0005, 该文件峰值帧 RMS 的 3%)`，填补不超过 100 ms 的词内间隙，并要求
播放保护期后至少存在配置要求的最短语音。真实测试集没有独立近端轨，因此只做
场景级正负样本判断，不报告精确起点延迟。

## 固定评测数据集

VoiceBox-Assistant-Evaluation-v2仅用于VoiceBox.Assistant.AecChallenge.Benchmark进行评测，与DTD训练毫无关系。

不要在每次评测时随机扫描完整数据集。使用下面的命令生成一个版本化的小型
评测集；默认使用同盘硬链接，不复制约 3 GB 的逻辑音频内容：

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --prepare-dataset `
  --aec-root E:\Projects\render-aware-dtd\datasets\AEC-Challenge `
  --kws-root E:\Projects\kws-train\datasets `
  --but-rir-root E:\Projects\render-aware-dtd\datasets\BUT_ReverbDB_rel_19_06_RIR-Only `
  --but-libri-root E:\Projects\render-aware-dtd\datasets\BUT_ReverbDB_rel_19_06_LibriSpeech-Only `
  --wenet-root E:\Projects\render-aware-dtd\datasets\WenetSpeech4TTS `
  --prepared-output E:\ML\Audio\benchmark\VoiceBox-Assistant-Evaluation-v2 `
  --materialization hardlink
```

按当前生成器重建的 v2 manifest 包含 1,349 个 Phase 1 AEC case、
189 个中文语音资产（其中 63 个来自磁盘上实际存在的 WenetSpeech4TTS wav/txt
配对）、108 个 BUT RIR（54 组合法几何）、90 个房间环境声/MUSAN 音乐与噪声/DNS
噪声，以及 164 个 Phase 2 端到端 case。`manifest.json` 的 `clusterId` 是统计
独立单位；同一 session、原始 utterance 或其噪声变体不能重复计数。

schema 2.0 为每个端到端 case 分别记录 `expectHeard`、`expectDuck`、
`expectAccepted`、`expectPlaybackStop` 和 `expectPlaybackResume`，并记录
`oracleSpeechIntervals`、`speakerPolicy`、`split`、`geometryId`、源文件 SHA-256、
生成器 commit 和参数。读取器仍兼容已有 schema 1.0 manifest，但生成器只写 v2。

BUT LibriSpeech-only 只保留少量待校准素材，当前不进入正式分数，因为目录中没有
对应的原始 LibriSpeech render，不能构造合法的 AEC reference。补齐原始 render
后可扩展为真实房间纯回声 suite。

先验证评测器自身：

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- --self-test
```

Phase 1 smoke：

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --dataset E:\ML\Audio\benchmark\VoiceBox-Assistant-Evaluation-v2 `
  --suite prepared-smoke `
  --verbose
```

Phase 1 正式小型集：

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --dataset E:\ML\Audio\benchmark\VoiceBox-Assistant-Evaluation-v2 `
  --suite prepared-phase1 `
  --callback-samples 256 `
  --output src/TestResults/barge-in-prepared-phase1.json `
  --enforce-gates
```

原始 AEC Challenge 的 `standard`、`device-sample` 等 suite 仍然可用。

也可以使用统一入口：

```powershell
.\scripts\evaluate-barge-in-aec-challenge.ps1 `
  -DatasetRoot F:\Temp\AEC-Challenge `
  -Suite standard `
  -EnforceGates
```

`standard` 包含 synthetic 的 500 条 test split，以及 2021 clean/noisy 的双讲
正样本和远端单讲负样本。`icassp2022` 会把 48 kHz 输入按生产采样率重采样到
16 kHz。默认 callback 为 256 samples，与当前 VoiceAssistant Demo 的
`MaxFrameSize` 一致；可用 `--callback-samples 160,256,320` 做回调粒度鲁棒性矩阵。

## 设备样本

`device-sample` 从以下真实录音源按 `corpus/scenario/label` 确定性均衡抽样：

- `datasets/real`、`datasets/real_doubled`：Windows PC；
- `datasets/fullband` 或 `datasets/real_doubled_fullband`：下载并解压后自动发现；
- `datasets/fullband_mobile` 或 `datasets/real_doubled_fullband_mobile`：
  Android mobile，下载并解压后自动发现。

默认每层抽 50 条；可用 `--limit-per-stratum` 调整。文件名前缀作为匿名
`deviceSession`，只用于分组和追溯，不宣称它等于公开设备型号。

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --dataset F:\Temp\AEC-Challenge `
  --suite device-sample `
  --output src/TestResults/barge-in-device-sample.json
```

设备类别分别产生覆盖、召回和远端单讲误触发门禁。Windows 或 Android 缺失、
正负样本少于门禁要求时状态为 `inconclusive`，不能用另一类设备的结果代替。
公共 real 数据中缺少 mic/lpb 配对的条目会被跳过，但数量写入
`configuration.skippedIncompletePairs` 并打印到控制台。

当前 AEC Challenge 数据目录已经包含 Windows 和 Android fullband 音频；Android
只有双讲，因此 Android 专属远端误触发率仍会保持 `inconclusive`。

## 报告和门禁

每次运行产生：

- JSON（schema `2.1`）：代码与数据集 Git commit/worktree 状态、配置、运行环境、
  总体/场景/SER/噪声/设备类别分层指标、门禁和全部逐样本结果；
- `*.cases.csv`：便于按 SER、噪声、movement、判定原因和概率筛选失败样本。

默认建议门禁按 callback 粒度分别计算，不能把同一音频在多个 callback 配置下的结果
当成独立样本扩充置信度。准备集有足够 cluster 时，召回率使用确定性的
session/utterance cluster bootstrap，并以 Wilson 区间作为小样本和完美结果的保护边界：

| 指标 | 门槛 |
|---|---:|
| 总体及每个正样本场景的召回率 95% Wilson 下界 | `>= 90%` |
| 远端单讲误触发率的单侧 95% Poisson 上界 | `<= 1 次/小时` |
| synthetic 合法起点到判定的右删失 P95 | `<= 700 ms` |

延迟统计把漏检和过早误触发视为“延迟未知且大于录音窗口”，不会只在成功样本上
报告一个偏乐观的 P95。若至少 5% 的 oracle 正样本被删失，P95 为无界并判失败。

召回率点估计达到门槛但置信下界未达到，或零误触发但负样本暴露时长不足以证明
目标误触发率时，门禁为 `inconclusive`，不会把“不够数据”误报为通过。以
`1 次/小时` 为目标且观测到零次事件时，至少需要约 3 小时独立负样本暴露才能让
单侧 95% 上界低于目标；重复同一音频不增加独立证据。AEC Challenge 自身暴露
不足时，需要补充真实设备的长时远端单讲回放集。

基础样本量不足时门禁同样是 `inconclusive`，不会把 smoke 当成正式结论。
`--enforce-gates` 会在任一门禁失败或样本不足时返回非零退出码。

## 基线与候选严格对比

完整基线运行及环境、模型哈希、manifest、源码差异证据可通过以下入口一起保存：

```powershell
.\scripts\evaluate-assistant-baseline.ps1 `
  -OutputRoot E:\Projects\voicebox\artifacts\baseline-20261012-current-model `
  -Provider cuda
```

该入口先构建并执行评测器自检，再串行执行 `prepared-phase1` 全量集
（callback 256）和 `nightly-e2e` 全部 164 个 case（manifest 内 callback 160/256/480）。
输出目录中已有正式报告时拒绝覆盖；重跑必须指定新目录。门禁失败仍保存完整报告，
分别在 `acoustic.status.json` / `e2e.status.json` 中记录退出码，并校验模型和 manifest
在运行前后未发生变化。具体基线结论见
[2026-10-12 基线记录](../../../docs/benchmarks/render-aware-dtd-baseline-20261012.md)。

当前 E2E 报告使用 schema `3.0`：延迟从合法近端触发时刻计算，分位数仅统计有观测的
事件，同时保留漏检/删失数；提前误触发不进入延迟分位数。Phase 1 仍使用 schema
`2.1` 的右删失延迟协议。比较时必须保持各自协议及共享模型、数据、执行后端一致。

声学和端到端候选评测完成后，使用同一入口对冻结基线做逐 case、逐 slice 和门禁对比：

```powershell
dotnet run -c Release --no-build `
  --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --mode compare `
  --baseline-acoustic src/TestResults/baseline/acoustic.json `
  --candidate-acoustic src/TestResults/candidate/acoustic.json `
  --baseline-e2e src/TestResults/baseline/e2e.json `
  --candidate-e2e src/TestResults/candidate/e2e.json `
  --baseline-evidence src/TestResults/baseline `
  --candidate-evidence src/TestResults/candidate `
  --output src/TestResults/candidate/comparison.json `
  --csv src/TestResults/candidate/comparison.csv `
  --enforce-comparison
```

对比器要求数据集 manifest/hash、case 与真值、callback 集合、命令、运行环境与后端、
统计协议及共享 ASR/TTS/VAD 模型一致；DTD 算法身份、配置和算法模型 hash 允许变化。
它检查 overall 以及 callback、corpus/scenario、SER、noise、geometry、render-gap、
device 等全部已序列化分层。新增逐 case 错误、提前触发、远端单讲误触发、指标回退、
门禁降级或新增 inconclusive 均判为失败。`--enforce-comparison` 在状态为 `Regressed`
或 `Incomparable` 时返回退出码 1；未指定 `--csv` 时会在 JSON 旁生成同名 `.csv`。

## 中文端到端运行

需要真实 VoxCPM TTS 和 Qwen3 ASR 模型路径。`--provider` 只控制 Qwen3-ASR 与
FireRed VAD；VoxCPM 始终使用 CPU，避免 8 GB 显存同时驻留 Qwen3-ASR 和 TTS。
ASR 推荐使用 CUDA；CPU 也可运行，但 1.7B ASR 会显著慢于实时：

```powershell
dotnet run --project src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark -- `
  --mode e2e `
  --dataset E:\ML\Audio\benchmark\VoiceBox-Assistant-Evaluation-v2 `
  --suite release-e2e `
  --asr-model D:\ML\Audio\voicebox_models\v1\asr\qwen3\qwen3-asr-1.7B-onnx-fp16 `
  --tts-model D:\ML\Audio\voicebox_models\v1\tts\voxcpm\voxcpm-0.5b-onnx-int8 `
  --provider cuda `
  --output src/TestResults/barge-in-e2e.json `
  --enforce-gates
```

可用 suite 为 `smoke`（9）、`pr-e2e`（31）、`release-e2e`（82）和
`nightly-e2e`（164）。每个 case 使用 CPU 上的真实 VoxCPM TTS；每个虚拟声卡 callback 从
Assistant 的播放 FIFO 读取当时仍在播放的 render，应用当前 duck gain、render gap、
时钟漂移、轻度扬声器非线性和流式 BUT RIR，再叠加近端中文语音及环境噪声生成
microphone。`ClearQueue()` 会让未来 render 归零，但 RIR 尾音继续传播。

E2E 使用与 Demo 一致的 FireRed VAD 和 `VoiceAssistantOptions` 生产默认值，只将
`BargeIn.Enable` 打开。报告分别给出 `Heard`、duck、ASR Accepted、控制层
`ClearQueue`、实际可听停止及恢复结果。控制停止延迟从近端语音起点计到
`ClearQueue`，可听停止延迟继续计到 RIR 回声连续 60 ms 低于门槛；另行报告的
可听尾音延迟只计算从 `ClearQueue` 到该静音点的声学尾部。

## 解释边界

- Phase 1 测的是 `Heard` 声学判定，不使用 AECMOS 代替打断事件。
- Phase 2 是可复现的离线声学数字孪生，不等价于真实扬声器、麦克风、AGC、
  削波和时钟硬件；正式发布前仍需少量 hardware-in-loop 校准。
- BUT RIR 已补偿扬声器到麦克风的传播延迟，因此 Phase 2 额外加入显式延迟；
  这也是为什么不能直接把 RIR 当作完整设备模型。
- Phase 1 默认不启用 WebRTC NS，使用 `--noise-suppression` 可测 AEC+NS 变体。
  当前 E2E 固定启用 WebRTC NS、关闭 AGC2；这两个模式的采集链配置分别记录在基线证据中。
- 结果只对报告中记录的 Git commit、参数、callback 粒度和数据子集有效。
