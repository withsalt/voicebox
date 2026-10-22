# Render-Aware DTD 基线（2026-10-12）

本记录对应当前集成模型的完整 Phase 1 与 E2E 重跑，作为后续 DTD 训练候选的回归基准。两套评测均完成全部 case 且没有执行错误；由于性能门禁未通过，benchmark 进程以退出码 1 结束。这表示基线暴露了待改进指标，不表示评测未完成。

完整 JSON、逐 case CSV、运行日志、环境信息、模型哈希和冻结 DTD 文件保存在本机 `artifacts/baseline-20261012-current-model`。`artifacts/` 被 Git 忽略，因此仓库中的本记录保存可读摘要，完整机器证据保留在该工作区。

## 输入与运行环境

- VoiceBox commit：`fd9f9b5183346e06d184c1a3fa2b5ed98b959484`
- DTD：`v6 candidate-01`，validation-frozen epoch 147；运行阈值 `0.55`，CPU 单线程；ONNX SHA-256：`3EB35D158CCAD47A072BFD57DBD99467E522F41A4D9FF414E4C6399F6245F156`
- 数据集：`VoiceBox-Assistant-Evaluation-v2`；manifest SHA-256：`0C9219CEE1F14FD4046A20C483DB6F91BB205F53093181240716260D0B8F0666`
- Phase 1：schema 2.1、callback 256、prepared-phase1，全量 1,349 cases
- E2E：schema 3.0、nightly-e2e，全量 164 cases；Qwen3 ASR / FireRed VAD 使用 CUDA，VoxCPM 使用 CPU（seed 23）
- 主 AEC：WebRTC APM CPU；Phase 1 关闭 NS，E2E 开启 NS，AGC2 关闭
- 主机：Windows 11 Enterprise LTSC 10.0.26300、.NET SDK 10.0.401、AMD Ryzen 9 9950X、NVIDIA GeForce RTX 5070 Ti 16 GB（驱动 616.92）
- 实际评测时间：Phase 1 360.3 秒，E2E 221.4 秒；RTF 分别为 `0.0187` 和 `0.1223`
- manifest 校验稳定，运行期间没有检测到模型文件变化

## Phase 1

| 指标 | 结果 |
|---|---:|
| Case / 正负样本 | 1,349 / 461 正、888 负 |
| TP / FN / FP / TN | 434 / 27 / 33 / 855 |
| Recall | 94.14%（95% CI：91.61%–96.26%） |
| Precision / specificity | 92.93% / 96.28% |
| 远端单讲误触发 | 7.903 次/小时；单侧 95% 上界 10.568 次/小时，负样本暴露 4.175 小时 |
| 合法近端起点至判定延迟 | P50 396 ms，P95 1,973 ms（95 个观测、1 个删失） |

全局 recall 门禁通过。未通过项为远端误触发率（上界目标 `<= 1 次/小时`）、合法起点至判定 P95（目标 `<= 700 ms`）和 noisy / double-talk-with-movement recall（86.67%，目标 `>= 90%`）。多个设备及数据切片因样本数量或置信区间不足标记为 `inconclusive`。

## E2E

| 指标 | 结果 |
|---|---:|
| Case / 独立 cluster | 164 / 141（87 正、54 负） |
| Heard / accepted / playback-stop recall | 85.06%（74/87） |
| Heard latency | P50 387 ms，P95 1,475 ms；13 个正样本删失 |
| 控制停止 / 设备 drain 完成延迟 | P95 1,713 ms / 1,478.5 ms |
| Echo-only heard / accepted | 6 / 6；false accept 为 29.60 次/小时（负样本暴露 0.203 小时） |
| Early false trigger | 3 |
| Aggregate CER / exact match | 46.25% / 20.69% |

Heard recall 门禁通过（目标 `>= 85%`），设备 drain 完成延迟门禁通过（目标 `<= 2,035 ms`）。未通过项为 Heard P95（目标 `<= 1,000 ms`）和 false accept（目标 `<= 1 次/小时`）。E2E 的 duck 指标仅跟踪，不作为门禁；当前控制链使用停止播放而非衰减播放。

## 原始证据

- [Phase 1 报告](../../artifacts/baseline-20261012-current-model/acoustic.json) · [逐 case CSV](../../artifacts/baseline-20261012-current-model/acoustic.cases.csv) · [运行日志](../../artifacts/baseline-20261012-current-model/acoustic.log)
- [E2E 报告](../../artifacts/baseline-20261012-current-model/e2e.json) · [逐 case CSV](../../artifacts/baseline-20261012-current-model/e2e.cases.csv) · [运行日志](../../artifacts/baseline-20261012-current-model/e2e.log)
- [运行状态与输入稳定性](../../artifacts/baseline-20261012-current-model/run-status.json) · [模型文件哈希](../../artifacts/baseline-20261012-current-model/model-files.sha256.csv) · [环境与后端记录](../../artifacts/baseline-20261012-current-model/execution-backends.txt)

后续候选必须在相同 manifest、共享 ASR/TTS/VAD 模型、执行后端和统计协议下评测，并通过 benchmark 的 compare 模式与以上两份 JSON 做逐 case、逐 slice 和门禁对比。当前 DTD 包元数据将该模型标记为 `NOT ACCEPTED / INCOMPLETE`；本结果是性能基线，不是发布验收结论。
