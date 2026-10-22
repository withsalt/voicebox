# Render-Aware DTD 训练档案（2026-10-19 归档）

> 范围：R0–R34（2026-10-12 ~ 2026-10-19），含 A–H 前史之外的完整训练/评测记录。
> 主台账：[render-aware-dtd-round-ledger.md](render-aware-dtd-round-ledger.md)（R0–R34 汇总表 + 各轮详情）。
> 决策包：[render-aware-dtd-decision-pack-20261019.md](render-aware-dtd-decision-pack-20261019.md)。
> 机器可读轮次记录：`artifacts/dtd-rounds/<round>/round.json`（所有轮次的假设、配方、指标、判定）。

## 1. 最终结论（2026-10-19）

1. **达标未达成**：在"模型 + DTD 自身配置"改动面内、冻结判决栈（AEC/融合窗/累加器/ASR/TTS/VAD）与**单模型推理**约束下，没有任何单模型点同时通过 A1/A2 + B1–B5 + C 判据。
2. **唯一声学全过的配置**是 R11 融合链（PASS 7/11），但**多模型融合被用户禁止**（推理成本），仅作诊断证据。
3. **两个最佳单模型候选**（均 PASS A3/B1/B5/C1/C2）：
   - **R28 ep9**：声学 recall .9501/.9262、FP 24（**5.67/h，U95 7.97**）、P95 2172、early 0、B3 .7032；e2e heard .8621、**falseAccept 9.87/h**、heardP95 2798、drain 2801。A1 Regressed（7 flips = 5 新 FP + 2 新 FN）。
   - **R23 ep12**：声学 .9544/.9314、FP 29（6.88/h）、P95 2172、early 0；e2e heard .8851、FA 14.80、heardP95 2792、drain 2794。A1 Regressed（7 flips = 7 新 FP、0 新 FN）。
4. **延迟专项（用户 2026-10-19 指令）**：唯一有效机制是 **onset-aligned 决策蒸馏**（纠正教师的晚触发时序）：
   - R30：e2e heard P95 **2798→2345**、drain 2801→2352、P50 回到基线 396、**B3 首次严格增益 .7438**；代价 FP 14.66/h。
   - R31（+难负样本）：FP **9.56**、P50 422、P95 2148、B3 .7438。
   - 进一步加压（R32 onset 1.0）能把声学 P50 压到 368 / P95 2057，但 **e2e 尾反恶化到 2720**、FP 20.75；focal（R33）与中 SER 正例上采样（R34）均无改进。
   - **结论**：训练侧 e2e 延迟下限观测 ≈ **2345ms @ FP 14.7**（R30）或 ~2400 @ FP 9.6（R31），仍高于基线 1475。回归由 **1–2 个案例**主导（e2e-069 单例 +2340ms），中位延迟仅比基线高 30–80ms；判决栈自身（acceptance 180–260ms）在模型间完全一致，不是瓶颈。
5. **已量化的结构墙**：
   - **B1↔B3 严格单轴**（低 FP 端 R23/R28 vs B3 端 R26 .7868）
   - **B2 P95 由固定 synthetic 尾例钉死**（2144–2172；基线 1973）
   - **A1 零 case 翻转对任何独立模型不可达**（历史最少 4–7 flips）
   - 所有可及杠杆（数据/标签/特征/通道/蒸馏/锚权/阈值/校准）均落回同一精度↔召回/延迟轴

## 2. 候选模型清单

| 候选 | 权重/ONNX | 声学 | e2e | 定位 |
|---|---|---|---|---|
| 基线（随包 e147） | `src/VoiceBox.DTD.RenderAware/runtimes/models/dtd/render-aware/render_aware_dtd.onnx`（sha256 3EB35D15…） | .9414/.9161、33（7.903/10.568）、396/1973、early0、B3 .7032 | heard .8506/P95 1475、FA 29.60、drain 1478.5 | 冻结基线 |
| **R28 ep9** | `artifacts/dtd-rounds/r28/training/output/checkpoints/epoch-009.pt`；ONNX `artifacts/dtd-rounds/r28/onnx/9/` | .9501/.9262、**24（5.667/7.973）**、450/2172、early0、B3 .7032 | heard .8621/P95 2798、**FA 9.87**、drain 2801 | 低 FP/e2e-FA 最佳 |
| **R23 ep12** | `artifacts/dtd-rounds/r23/training/output/checkpoints/epoch-012.pt`；ONNX `artifacts/dtd-rounds/r23/onnx/12/` | .9544/.9314、29（6.880/9.380）、416/2172、early0、B3 .7032 | heard .8851/P95 2792、FA 14.80、drain 2794 | 召回/heard 更高 |
| R30 ep5（延迟优先） | `artifacts/dtd-rounds/r30/training/output/checkpoints/epoch-005.pt`；ONNX `r30/onnx/5/` | .9544/.9314、60（14.66）、396/2144、early0、B3 **.7438** | heard .8736/P50 431/**P95 2345**、drain 2352、FA 29.6 | e2e 延迟最佳 |
| R31 ep5（延迟/FP 平衡） | `r31/training/output/checkpoints/epoch-005.pt`；ONNX `r31/onnx/5/` | .9523/.9288、40（9.56）、422/2148、early0、B3 .7438 | 未跑（预计 ~2400） | 平衡点 |
| R17 ep0（P95 观察点） | `r17/training/output/checkpoints/epoch-000.pt` | .9566/.9339、45（10.94）、?/**1961**、early0 | 未跑 | 唯一非融合 P95 击穿 |

所有候选均为**单模型 2ch**、契约与基线一致（输入 mic+render、state [3,1,288]），部署零额外推理成本（R18/R19 的 4 输入/状态扩展模型仅存在于对应轮次）。

## 3. 数据资产（E:\Projects\render-aware-dtd\datasets）

| 资产 | 说明 | 状态 |
|---|---|---|
| render-aware-dtd-v5 | 基线家系（48,000 train + 1,500 val，candidate-01 同源） | 保留 |
| render-aware-dtd-v8r1 | 设备族扩展池（26,138 行）物化 | 保留（R11 系基座） |
| render-aware-dtd-v8r2/3/4 | TTS far 域剂量/内容（R7–R9，均否决） | 可删（配方留 round.json） |
| render-aware-dtd-v8r5 | v8r1 + 真实双讲弱标签 15%（R11） | 保留（历史突破轮） |
| render-aware-dtd-v8r6 | 同 30% 剂量（R12 否决） | 可删 |
| render-aware-dtd-v8r7 | **干净弱标签子集（ERLE≥5dB，3,610 行）**，R20–R34 全部轮次的基座 | **保留（现役）** |
| render-aware-dtd-v8r10 | v8r7 + 中 SER 带噪 mixture 行复制（R34 否决） | 可删 |
| real-doubletalk-corpus | R11 弱标签语料（8,434 段 + 元数据） | 保留 |
| tts-online-corpus(-v2)/tts-qwen3-corpus | TTS far 域合成语料（R7–R9） | 保留（可删） |
| _v8r1-regen | manifest-only 重生成（数据事故恢复用） | 可删 |

蒸馏/锚定目标（artifacts）：
- `r16/distill-targets`：R11 链软概率（v5，49,500）
- `r17/anchor-targets`：e147 保持概率（v5）
- `r22/teacher-targets`：R11 链概率（v8r7，49,500）
- `r22/decision-targets`：链决策二值（≥0.55，v8r7）
- `r30/onset-aligned-targets`：**onset 对齐决策目标**（填补 1,022,434 帧 + 救回 596,817 帧）

## 4. 工具（scripts/render-aware-dtd/rounds/，随仓库）

数据：`build_eval_exclusions.py`、`build_v8_real_pools.py`、`build_v8r2_tts_pools.py`、`screen_v8_pool_rows.py`、`verify_v8_dataset.py`、`synthesize_tts_corpus.py`、`synthesize_qwen3_corpus.py`、`label_real_doubletalk.py`、`inject_real_doubletalk.py`、`replay_injection_manifest.py`、`materialize_branches.py`
教师/蒸馏：`score_distill_teacher.py`、`binarize_teacher_targets.py`、`build_onset_aligned_targets.py`
训练：`guard_training.ps1`、`watch_training.py`、`select_epochs.py`、`sweep_threshold.py`、`verify_e147_reproduce.py`
评测：`export_and_eval.ps1`（含 `-AecChannelSource`/`-ProbabilityTemperature`）、`run_e2e_and_compare.ps1`（`-AcousticCompareOnly`/`-SkipEnvironmentStability`）、`check_acceptance.py`、`check_environment_stability.ps1`、`swap_and_eval.ps1`
融合（诊断）：`export_fused_veto_onnx.py`、`export_fused_union_onnx.py`、`analyze_union_rule.py`、`analyze_veto_rule.py`、`collect_dual_scores.py`
探针：`probe_coherence.py`（相位敏感互谱相关负结果）

训练侧新增能力（本轮次归档新增）：
- `train_dtd.py`：`--init-checkpoint` / `--freeze-frontend` / `--baseline-anchor-weight` / `--history-frames` / `--correlation-lags` / `--positive-focal-gamma`
- `dtd_model.py`：可选因果跨时间相关前端（R19）、state 扩展、`StreamingDtdWrapper.probability_temperature`
- `materialized_v5_dataset.py`：`DTD_ANCHOR_TARGETS` / `DTD_DISTILL_TARGETS` / `DTD_AEC_SOURCE=apm` sidecar / `_labels.npy`
- `export_dtd_onnx.py`：4 输入契约、`aec_channel_source` metadata、`--probability-temperature`
- C# 运行时：`IRenderAwareDtdService.HasAecOutputInput/PostAecChannelSource`、内容精确 post-AEC 读取+hold、`SpectralAecProxy`

## 5. 轮次工件索引

`artifacts/dtd-rounds/<round>/` 的标准布局：`round.json`（判定）、`training/output/{history.json,checkpoints/,best.pt,last.pt}`、`selection/epoch-selection.json`、`onnx/<epoch>/`（ONNX+contract+parity）、`acoustic-<epoch>/{acoustic.json,cases.csv}`、`e2e/{e2e.json,cases.csv}`、`compare/{comparison-acoustic.json,comparison-full.json}`、`acceptance.json`。
逐轮索引见 `artifacts/dtd-rounds/ARCHIVE-MANIFEST.md`。

## 5.5 磁盘占用（2026-10-19 实测）

- `artifacts/dtd-rounds/` ≈ **214 GB**（主要来自每轮的隔离运行时副本 `runtime-*/`、checkpoints 与 ONNX）
- `datasets/render-aware-dtd-v8r7` ≈ 37 GB（其余 v8r1–v8r10 各 ~37 GB，多为 hardlink 共享）
- 可清理候选（需用户确认）：各轮 `runtime-<epoch>/`（评测用二进制副本，可重建）、已结案且不会再用的 `v8r2/v8r3/v8r4/v8r6/v8r10`、`_v8r1-regen`

## 6. 未决事项

- 交付点未定：R28 ep9（FP/e2e-FA 最深）或 R23 ep12（召回/heard 更高）或延迟优先的 R30/R31。
- 若要 e2e P95 ≤1475，训练侧杠杆已耗尽；需讨论冻结判决栈（累加器/证据窗）或接受折衷。
- 打包替换随包模型是独立步骤，需用户确认。
- 仓库改动未提交（R14–R34 的全部代码/配置/工具/台账）。
