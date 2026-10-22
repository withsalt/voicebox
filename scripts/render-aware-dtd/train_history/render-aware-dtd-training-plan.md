# Render-Aware DTD 长期训练优化方案

- 制定：2026-10-12；状态：R0 基建已完成并自证（见 `scripts/render-aware-dtd/rounds/README.md` 与 `artifacts/dtd-rounds/R0-selftest/EXPERIMENT.md`），R1 待启动
- 基线记录：[render-aware-dtd-baseline-20261012.md](render-aware-dtd-baseline-20261012.md)；基线证据：`artifacts/baseline-20261012-current-model/`
- 轮次台账：[render-aware-dtd-round-ledger.md](render-aware-dtd-round-ledger.md)
- 历史结论（A–H 八轮全部否决）见记忆 `dtd-round-abc-optimization`；本方案建立在"设备族覆盖"新诊断上（见 §2.3）

## 1. 目标与达标定义（2026-10-12 用户确认）

**总目标**：训练新的 render-aware DTD 模型，替换 `VoiceBox.DTD.RenderAware` 随包模型后，在两套评测上满足下述达标定义。评测必须使用与基线完全相同的协议：同 `VoiceBox-Assistant-Evaluation-v2` manifest、同命令、同后端（见基线 `commands.txt`）。

**达标定义（loop 的停止条件）**——三件事同时成立：

- **A. 无回退**（2026-10-12 R0 实验修订，原"compare 退出码 0"表述被证不可达——e2e 层 CER/切片停止延迟运行间非确定，同模型重放两次即产生 20 条 e2e-slice finding，详见 `artifacts/dtd-rounds/R0-selftest/EXPERIMENT.md`）：
  - **A1（机器硬门）**：声学层 `--mode compare --acoustic-only --enforce-comparison` 退出码 0（该层已证逐位可复现）；
  - **A2**：完整 compare（acoustic+e2e）中无"严重回退"——任何 scope 非 `e2e-slice` 的 finding（含 e2e 决策级逐 case、门禁降级、声学层）均为失败；`e2e-slice` 仅豁免实测噪声指标 {aggregateCharacterErrorRate, stopLatencyP95Ms, audibleStopLatencyP95Ms, audibleTailLatencyP95Ms}，豁免清单扩容需新证据；
  - **A3**：acoustic/e2e errorCount 均为 0。
- **B. 5 个当前未过项全部严格优于基线**：

  | # | 项目（报告字段） | 基线段 | 要求 |
  |---|---|---|---|
  | 1 | acoustic `farend-false-triggers-per-hour` | 点 7.903 / U95 10.568 | 点与 U95 双降 |
  | 2 | acoustic `eligible-onset-to-decision-p95-ms` | 1,973 | 严格下降 |
  | 3 | acoustic `recall:test_set_noisy/doubletalk_with_movement` | L95 0.7032（点 86.67%） | L95 与点严格上升 |
  | 4 | e2e `heard-latency-p95-ms` | 1,475 | 严格下降 |
  | 5 | e2e `false-accept-per-hour` | 29.60（6 个 echo-only heard） | 严格下降 |

- **C. 其余项目不劣**：包括已通过项不降级（acoustic 全局召回 L95 0.9161、synthetic 召回、e2e heard recall 85.06%、e2e drain P95 1478.5），0 错误/0 早触发保持 0。A 的 compare 已覆盖大部分；判定脚本按 §3.9 汇总。

**改动边界（用户确认）**：训练数据、模型结构、损失函数、导出阈值等 **DTD 自身配置** 可改（阈值只允许用验证集选择，禁止看评测集）；AEC、fusion 窗口、accumulator、ASR/TTS/VAD 以及其它运行时全部冻结。扩展 ONNX 输入契约需另行批准（本方案含兜底条目 R6）。

**北极星（非硬性停止条件）**：全部门禁通过（FP≤1/h、P95≤700ms/1000ms、各切片召回 L95≥90%、e2e falseAccept≤1/h）。

## 2. 现状诊断（策略依据）

### 2.1 基线事实

- 模型：v6 candidate-01 / epoch-147，ONNX sha256 `3EB35D15…`（打包于 `src/VoiceBox.DTD.RenderAware/runtimes/models/dtd/render-aware/`）。
- 训练：`E:\Projects\render-aware-dtd\artifacts-v6\candidate-01`，v5 数据集（48,000+1,500 条 8s/16k 物化样本）；混合权重 narrow 0.4 / fullband 0.2 / 预生成物化样本 0.2 / aec_real 0.2；1,656,577 参数（80 bands、288/288、3 层 GRU）。
- 评测结果与失败门禁见 §1.B 表；acoustic TP/FN/FP/TN = 434/27/33/855。

### 2.2 八轮历史结论（为什么这次做别的）

A–H 八轮：数据分布增广（对齐失配/movement/far-only 噪声重平衡）、损失函数（event loss）、容量（宽度/深度/bands/全扩）——**全部被基线支配或 benchmark 判回退**，且失败形状同构：修好约 10 个漏检、必然新引入 21–27 个误报。单变量纪律与"以 acoustic 为准、e2e 对误报不灵敏"的教训继续沿用。

### 2.3 2026-10-12 新诊断（本方案的核心依据）

对基线逐 case 盘点 + 对评测集做源文件溯源（硬链接实证）后：

1. **33 个 acoustic FP 全部落在训练从未覆盖的语料/设备族**：test_set_icassp2022 = 18、test_set_noisy = 7、device_fullband_mobile = 6、device_fullband = 1、test_set_clean = 1。相反，训练池覆盖过的 device_real / device_real_doubled farend 切片 **0 FP**。
2. 27 个 FN 中 device_fullband_mobile 7、test_set_noisy 6、test_set_clean 5、device_real 4、device_real_doubled 2、device_fullband 2、synthetic 1 —— 漏检大头同样集中在未覆盖设备族（移动端全宽带）。
3. 96 个评测负样本（device_real / device_real_doubled 的 farend_singletalk 各 48）的源文件与 v5 far_only 训练池**同名同文件**（基线继承的既有重叠，见 §3.2 处理规则）；评测 doubletalk 正样本与训练池无重叠。
4. 可用而未用过的真实语料（未被任何评测 case 使用，可整段入池）：

   | 语料目录 | 规模 | 评测占用 | 说明 |
   |---|---|---:|---|
   | `real_doubled_fullband` | ~1250 sessions | 93 sessions | Windows 全宽带（device_fullband 族） |
   | `real_doubled_fullband_mobile` | ~2250 sessions | 92 sessions | Android 移动（FP/FN 双重大户对应族） |
   | `blind_test_set/clean+noisy` | ~200 sessions | 0 | 2021 盲测族 |
   | `blind_test_set_icassp2022` | ~150+ sessions | 0 | **与评测 test_set_icassp2022 同届、session 不重叠**，直指 18-FP 族 |
   | `blind_test_set_icassp2023` / `blind_test_set_interspeech2021` / `test_set_interspeech2021` | 各 ~160–240 sessions | 0 | 更多设备族 |

   （session 数以 §R0 精确统计为准；上述规模按目录文件数/每 session 9–10 文件估算。）
5. 不可入训练：评测已整占的 session —— test_set_icassp2022 的 300 条 farend、test_set/clean 与 noisy 的全部 session、device_fullband/device_fullband_mobile/device_real/device_real_doubled 的 369 个 deviceSession。**新增语料一律按 session 级排除**（见 R0 排除清单）。

### 2.4 由此确定的策略优先级

先补**设备族覆盖**（证据最强、机制最清晰的未试方向），再谈容量与损失；数据侧单变量推进，每轮以 benchmark compare 为最终裁决。

## 3. 每轮固定流水线

### R0 基建（先决条件，一次性）—— ✅ 2026-10-12 完成并自证

在 `scripts/render-aware-dtd/rounds/`（新目录，随仓库提交，防止再随 artifacts 清理丢失）落地：

| 件 | 职责 | 要点 |
|---|---|---|
| `build_eval_exclusions.py` | 枚举评测用源文件/session → `eval-exclusions.json` | 方法：manifest + 命名映射（corpus→源目录、scenario→文件名后缀，注意 fullband 用连字符命名），抽样用 `fsutil hardlink list` 实证；覆盖 phase1 全部 1349 case 与 phase2 资产（aishell/wenet/BUT RIR/DNS），后者与 v5 池的重叠需单独核验并记录 |
| `watch_training.py` | 训练监控 | **先复制 history.json 快照再读**（历史故障：与原子重命名冲突触发 WinError 5 使训练崩溃）；报 valLoss/AP/F1/recall@低FPR；过拟合判据 = valLoss 抬头而 AP 不跟着回落 |
| `guard_training.ps1` | 训练守护 | 固定 `initial_num_workers=4`（6 必崩）、监视 commit 上限（~79.5GB）与 error 1455，崩溃后从 `last.pt` 自动续训；日志必须写输出目录**之外**（preset 校验"全新目录必须不存在"） |
| `select_epochs.py` | 验证集选点 | 复用 `select_v5.py` 的 `v5_event_lexicographic_v1` 策略；输出三选点：anchor-最近点、早停 best.pt、AP-best；产出 `epoch-selection.json`。自证：能从 candidate-01 历史独立复现 e147 |
| `export_and_eval.ps1` | 导出+声学评测 | 对每个选点：`export_dtd_onnx.py`（parity 门禁 < 1e-6）→ 拷贝 benchmark 的 Release 输出目录到隔离 runtime 目录（**源码树打包模型永不改动**）→ 替换其中 `runtimes/models/dtd/render-aware/render_aware_dtd.onnx`+`.contract.json` → 直接运行拷贝出的 exe 跑 `prepared-phase1` |
| `run_e2e_and_compare.ps1` | e2e + 对比 | 对通过声学门的点跑 `nightly-e2e`（cuda，Qwen3-ASR + VoxCPM CPU）→ `--mode compare --enforce-comparison`（基线 JSON 固定为 `artifacts/baseline-20261012-current-model/` 的两份） |
| `check_acceptance.py` | 达标判定 | 读同一轮 acoustic/e2e/compare JSON，按 §1.A/B/C 输出 PASS/FAIL 明细与退出码，loop 用它做停止判断 |

再验证（R0 内完成）：`build_eval_exclusions.py` 输出对拍我 2026-10-12 的盘点（96 文件重叠、无法入训 session 计数）；`select_epochs.py` 对 candidate-01 复现 e147。

### 每轮流程 P1–P9

- **P1 立项**：写 `artifacts/dtd-rounds/<round-id>/round.json`：假设、**唯一变量**、完整配置、数据集配方与 seed、排除清单哈希、预登记的成功/否决判据、预登记风险点（用于事后归因）。
- **P2 数据集生成**：新脚本族（v8 系列，**禁止改 v5/v7 原件与其 config**；v7 已实证"关闭新开关与 v5 逐位一致"，v8 沿用该模式）。生成前断言排除清单；生成后跑泄漏检查（新 manifest 与 eval-exclusions 交集必须为空）+ 与 v5 对比报告（哪些行随池变化、计数一致性）。输出到 `E:\Projects\render-aware-dtd\datasets\render-aware-dtd-<round>`（E 盘不足时放 F，路径写进 round.json）。
- **P3 训练**：`guard_training.ps1` 拉起；230 epochs 预算（沿用 candidate-01 的早停 patience=30 / min_delta=2e-4）；训练中 `watch_training.py` 每 20–30 分钟检查一次。
- **P4 选点（validation-only，禁止看评测集）**：跑 `select_epochs.py` 产出 ≤3 个点（anchor-最近 / best.pt / AP-best）。判定：
  - 若全部点被 anchor **严格支配**（验证集事件指标劣于 e147 的 validation 值）→ 免评测 veto，记录后回到 P1；
  - 否则进入 P5。
- **P5 导出**：逐点 ONNX 导出 + parity 校验 + 记录 sha256。
- **P6 声学评测**：逐点跑 `prepared-phase1`（允许 `--enforce-gates` 退出码 1，完整报告已保存）。
- **P7 声学判定门**：
  - 任一点违反 C（有回退）→ veto 该点（记录后淘汰）；
  - 至少一点满足 B 中 1–3 项改善且无回退 → 该点进入 P8；
  - 全部点只满足"无回退但无改善"→ 记录为中性轮，按 §5 调整方向。
- **P8 e2e + compare**：对入选点跑 `nightly-e2e`，再跑两次 compare（acoustic-only 硬门 + full 记录，baseline = 基线 JSON）。A1/A2 通过且 B 项 4–5 改善 → **达标候选**；否则按结果细化归因（哪个切片回退、FP/FN 配对拆解；e2e 噪声豁免清单外的任何 finding 都要先解释成因）。
- **P9 记账**：更新台账（§4），更新记忆；按数据集保留规则处理。

### 单变量纪律

每轮相对"当前最佳"只改一个可解释的变量（数据池 / 结构 / 损失 / 工作点各算一类）。确需打包两个改动时，round.json 必须写明理由与不可拆性论证（如历史上 H 轮的"深度+bands"）。

## 4. 记录与产物规范

### 4.1 台账

`docs/benchmarks/render-aware-dtd-round-ledger.md`：一行一轮的汇总表 + 每轮一个详情小节。字段：轮次 ID、日期、假设/变量、数据集（路径+manifest 哈希）、训练（epochs、best epoch、验证要点）、选点与导出（epoch、ONNX sha256、parity）、acoustic 要点（5 项 + TP/FN/FP/TN）、e2e 要点（4 项）、compare 状态、判定（达标候选/回退/中性）、数据集去留、下一步。

机器可读副本：`artifacts/dtd-rounds/<round-id>/summary.json`（同样字段），供 loop 读取。

### 4.2 每轮目录结构（evidence）

```
artifacts/dtd-rounds/<round-id>/
  round.json               # P1 立项
  dataset/                 # manifest 哈希、generation_config、泄漏检查报告（音频本体在数据集目录）
  training/                # config、history.json 快照、console log、崩溃/续训记录
  selection/epoch-selection.json
  onnx/<epoch>/{model.onnx,contract.json,parity.json,sha256}
  runtime-<epoch>/         # 隔离评测运行时（含被替换的 ONNX，评测证据）
  acoustic-<epoch>/{acoustic.json,cases.csv,status}
  e2e/{e2e.json,cases.csv,status}
  evidence/                # 与基线同构的 compare 证据（commands/环境五件套/模型哈希/execution-receipt）
  compare/{comparison-acoustic.json,comparison-full.json,*.csv,status}
  acceptance.json
  summary.json
```

### 4.3 数据集保留规则（用户指定）

- 轮次结果**优于基线**（成为当前最佳）→ **保留数据集**（记录路径；E 盘不足时放 F 盘）。
- 其它情况 → 判定记录完成后可删除数据集本体；**配方永远保留**（生成脚本+config+seed+排除清单+git commit，确定性可再生成——v7 已实证逐位可复现）。

### 4.4 采纳（打包替换）是独立步骤

达标后不自动替换源码树模型。采纳 = 用户确认 → 替换 `src/VoiceBox.DTD.RenderAware/runtimes/models/dtd/render-aware/render_aware_dtd.onnx(+contract)` → 更新 MODEL.md → 跑 `scripts/evaluate-assistant-baseline.ps1` 重出正式基线 → 更新记忆与基线文档。

## 5. 轮次 backlog（优先级队列，按轮次结果动态调整）

### R1（首轮）设备族覆盖扩展 ★详设

- **假设**：FP/FN 集中在未覆盖设备族；把 §2.3.4 的真实语料单讲片段加入 real 池，可压低对应族的误报并改善移动端召回。
- **唯一变量**：真实语料池扩展（模型结构与全部训练/生成超参为 candidate-01 原值；权重 0.4/0.2/0.2/0.2 不变）。
- **配方**：
  - real 池（同时供物化生成的 `aec_real_echo`/`aec_real_near` 两条路径与 `RealSingleTalkDataset` 分支）新增：`real_doubled_fullband`、`real_doubled_fullband_mobile`、`blind_test_set/clean|noisy`、`blind_test_set_icassp2022|icassp2023|interspeech2021`、`test_set_interspeech2021` 的 farend singletalk（→ far/echo 池）与 nearend singletalk（→ near 池）片段；
  - 全部按 session 级排除评测已占用 session；**不得新增任何与评测文件同文件的池条目**；
  - 既有 v5 池与评测的 96 文件重叠**原样保留**（继承基线条件，避免引入第二个变量；在 round.json 与台账中明示）；
  - 验证集保持与 v5 完全一致（保证 anchor 可比的选点协议）；新语料只进训练池。
- **训练**：seed 43、230 epochs、其余同 candidate-01。
- **预登记风险**：(a) 新族负样本可能把 FP 压向下一批未见族（2021 test_set 8 个 FP 所在族不可训练，评测已整占）；(b) 池扩容改变 real_echo 子集构成，可能轻度影响既有切片；(c) 移动端 near 片段音质差可能拉低 synthetic 切片表现。
- **成功判据**：P7 通过且 B 项 1（FP/h）改善；理想情况同时改善项 2–3。若仅"无回退无改善" → R2 前先复查池构成是否被大池稀释，必要时按族分层采样权重。

### R2 容量叠加（R1 胜者之上）

在 R1 胜者上叠加已验证的容量方向：**深度 3→4 + bands 80→128**（H 轮为八轮验证集最强，57% 参数换 E 轮级别效果；宽度维持 288 因单独扩容为负收益）。仍按"一个改动单元"记录论证。

### R3 未见设备 far-only 硬负样本强调

在胜者数据上，提高新设备族 far_only 负样本的采样权重/概率（族分层过采样或 `hard_far` 概率上调），单变量推进；若 R1 已解决 FP 则跳过。

### R4 工作点重选（DTD 自身配置，允许范围内）

模型稳定后，用验证集重新选择导出阈值（当前 0.55）与相关判决参数，目标直指 FP/h 与 P95 延迟的权衡；阈值选择全程 validation-only，改运行时默认前先在隔离 runtime 评测复核。**注意**：`RenderAwareDtdOptions.Threshold` 是运行时默认值（0.55），改它属于 DTD 配置改动，允许但在 compare 语义下属于"DTD 配置变化"。

### R5 真实双讲段利用（存档，需先解决标签）

real_doubled 系列的 `_doubletalk_mic.wav` 整段从未进过训练（无近端标签）。可行方向：用同 session 的 nearend_singletalk 做声纹/能量锚定做弱标签，或 session 级 MIL 损失。视为研究项，前四轮未见效后再启动。

### R6 输入契约扩展（兜底，需用户另行批准）

若数据与容量路线用尽仍未达标：扩展 ONNX 输入（如 AEC 输出/ERLE/延迟估计侧信号），同步改 C# 推理与接入。超出本方案默认授权，启动前单独确认。

### 每轮结束后的方向调整规则

- 达标 → 停止，进入采纳流程（§4.4）。
- 回退 → 按配对拆解（修好 X 个、新引入 Y 个）归因；若同 F 轮形状（修 N 引 >N）连续两轮出现，判定该维度耗尽，跳到下一优先级。
- 中性（无回退无改善）→ 检查池曝光/采样是否需要分层；必要时在下一轮附上"曝光计量"附录（每族实际进批次的 example 数）。
- 任何轮次结果优于基线即成为**新基线候选**（后续轮次相对它单变量推进；台账记录基线迁移）。

## 6. 风险与护栏

| 风险 | 对策 |
|---|---|
| Windows commit 上限（79.5GB, pagefile 16GB） | 4 workers 铁律；guard 监视 commit 与 error 1455；崩溃自动续训 |
| 监控脚本干扰训练（WinError 5 历史故障） | 一切读取先复制快照 |
| 训练资源预估 | ~113–120 s/epoch，230 epochs ≈ 6.5 h；单轮总预算（含生成/评测/对比）≤ 8 h |
| 磁盘 | E 余 ~413GB、F 余 ~890GB；每轮数据集 ~37GB；淘汰轮按 §4.3 清理 |
| 泄漏 | session 级排除 + 生成后交集断言；e2e 资产（aishell/wenet/BUT/DNS）重叠在 R0 核验并纳入排除清单 |
| 评测协议漂移 | 固定 manifest sha256（`0C9219…`）、固定命令、固定后端；每次评测保留 model-files.sha256 与 run-status |
| e2e 对误报不灵敏（八轮教训） | 永远以 acoustic 为先决层；e2e 只作 2/5 项裁决与 compare 补充 |
| 选点乐观偏差（E 轮教训) | 每轮必测 best.pt + anchor-最近点，必要时加 AP-best |
| 统计噪声（negative 暴露 4.175h / e2e 0.203h） | 判据按 §1 的点值与 U95 双指标；compare 逐 case 复核，不把单 case 波动当趋势 |

## 7. R1 执行清单

1. R0 全部件落地并自证（排除清单对拍、e147 复现）。
2. `round.json`（R1）：含 §5-R1 全部内容与预登记判据。
3. 生成 v8 数据集（E 盘，不足转 F），跑泄漏断言与 v5 对比报告。
4. 训练（guard），期间每 20–30 分钟 watch；崩溃自动续训并记录。
5. 训练结束 → 选点 → 导出 → 逐点 acoustic → 门 → e2e + compare → 台账 + 记忆。
