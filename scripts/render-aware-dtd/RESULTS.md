# Render-aware DTD results

## Candidate-2

日期：2026-07-31
数据集：`E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4`
基线：`scripts/render-aware-dtd/artifacts-enriched-v3/best.pt`
候选：`E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\best.pt`

candidate-2 从随机初始化开始训练 50 epochs。采样比例为 AEC narrow 45%、
AEC fullband 20%、Wenet 在线合成 15%、AEC real 20%，far-end 只使用
WenetSpeech4TTS。所有源文件运行时按文件头采样率读取、转 mono 并重采样到
16 kHz。

模型为 48 bands / projection 192 / GRU hidden 192 / 2 layers，共 510,337
参数，31.4928 MMAC/s。

### Strict non-regression gate

所有数值来自固定 `render-aware-dtd-v4` 测试套件；每个 delta 都是
`candidate - baseline`。门禁要求下列每一项 delta `>= 0`。

| Suite | AP baseline | AP candidate | AUC baseline | AUC candidate | best F1 baseline | best F1 candidate |
|---|---:|---:|---:|---:|---:|---:|
| AEC official | 0.977294 | 0.982480 | 0.976759 | 0.982247 | 0.924990 | 0.932669 |
| Wenet unseen | 0.587573 | 0.812241 | 0.772110 | 0.905678 | 0.537993 | 0.708020 |
| AEC real unseen | 0.985987 | 0.994216 | 0.992819 | 0.997289 | 0.933590 | 0.961689 |

| Suite | Recall @0.1% FPR | Recall @0.5% FPR | Recall @1% FPR | Recall @2% FPR | Recall @5% FPR |
|---|---:|---:|---:|---:|---:|
| AEC official baseline -> candidate | 0.383976 -> 0.437023 | 0.597525 -> 0.665762 | 0.693209 -> 0.751843 | 0.785574 -> 0.830612 | 0.891092 -> 0.912244 |
| Wenet unseen baseline -> candidate | 0.078613 -> 0.252798 | 0.131176 -> 0.354187 | 0.165396 -> 0.411068 | 0.212989 -> 0.475355 | 0.305066 -> 0.591862 |
| AEC real unseen baseline -> candidate | 0.655961 -> 0.748396 | 0.802349 -> 0.870179 | 0.855086 -> 0.916600 | 0.900363 -> 0.962514 | 0.958709 -> 0.996802 |

严格门禁结果：**PASS**。

### Candidate test reports

candidate-2 的标准测试报告：

- `E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\test_metrics.json`
- `E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\comparison.json`

checkpoint 工作点为 validation 选出的 frame threshold `0.53`。测试集上如果按
各 suite 独立扫描 threshold，AEC official / Wenet unseen / AEC real unseen 的
best F1 分别为 0.932669 / 0.708020 / 0.961689。该阈值不代表 Assistant 的
生产打断阈值；生产应基于连续概率和端到端 Heard 代价重新标定。

## Deployment artifacts

以下文件由 candidate-2 产生：

- `best.pt`
- `render_aware_dtd.onnx`
- `render_aware_dtd.contract.json`
- `render_aware_dtd.parity.json`
- `training_summary.json`
- `test_metrics.json`
- `comparison.json`

ONNX streaming parity（20 frames）：

- probability max abs error：`7.45e-8`
- probability mean abs error：`2.67e-8`
- state max abs error：`1.61e-6`
- state mean abs error：`1.99e-7`

## Interpretation and limits

candidate-2 在固定 official、Wenet unseen 和 AEC real unseen 套件上没有任何
指标下降，并明显改善了真实 AEC 的极低误触 operating points。这证明当前训练
配方和扩容模型满足离线 non-regression gate，但不等价于任意真实设备上的最终
Heard 体验：

- 还没有接入 `VoiceBox.Assistant`；
- 还没有跑 `VoiceBox-Assistant-Evaluation-v2`；
- frame-level detector 指标不包含 Assistant 的 fusion、score decay、360 ms
  accumulator 或 Heard 控制流；
- 生产阈值必须在接入后的音频节点和端到端 benchmark 上重新标定。
