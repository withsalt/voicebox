# VoiceBox.ASR.Qwen3.Benchmark

Qwen3-ASR streaming runtime 的回归 / 横评工具。读 AIShell-1 测试集，逐条跑 ASR，输出每条的 CER 与 RTF，并把汇总结果写成 JSON 落到 `src/TestResults/`。

支持四种典型用法：

| 场景 | 推荐参数 |
|---|---|
| 单模型一次性测精度/速度 | `--count 400` |
| 算法改前后回归检测 | `--compare <baseline.json>` |
| 多个模型横向对比 | `--models <dir1,dir2,...>` |
| AEC/DNR 前处理对 ASR 的影响 | `--matrix` |

---

## What it measures

Per-sample（每条样本）：

- `audio_seconds` / `inference_seconds` / `rtf`
- `cer`, `substitutions`, `insertions`, `deletions`, `ref_length`, `hyp_length`, `exact_match`
- `hypothesis` 与 `reference`（人工核对错例用）

Aggregate（一次跑的汇总）：

- `aggregate_cer` —— `(S + I + D) / total_ref_chars`，AIShell 标准 CER
- `exact_match_rate` —— 完全匹配（标点归一化后）样本占比
- `aggregate_rtf` —— `total_inference / total_audio`
- `warmup_seconds`, `wall_seconds`, `failed_count`

CER 比较前会做归一化（去标点、英文小写、Unicode NFC），逻辑见 `CerCalculator.cs:34`。

---

## Defaults

`Program.cs` 顶部常量；机器不同请用 CLI 覆盖。

| 项 | 默认值 |
|---|---|
| `--csv` | `E:\ML\Audio\dataset\speech_asr_aishell1_testsets\aishell1_test.csv` |
| `--model-dir` | `E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-float16` |
| `--onnx-native-dir` | `E:\ML\Audio\onnxruntime-win-x64-gpu-1.26.0\lib` |
| `--ep` | `cuda` |
| `--count` | `500` |
| `--seed` | `42` |
| `--chunk-size` | `30.0`（足够大让 AIShell 短句一次跑完，去掉 chunk 边界对精度的扰动） |
| `--max-new-tokens` | `256` |

---

## Available models

本机 6 份候选模型，都可以直接喂给 `--model-dir` 或 `--models`：

```
E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-float16
E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-float16-delta
E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-int8
E:\ML\Audio\models\Qwen3-ASR-1.7B-ONNX-float16
E:\ML\Audio\models\Qwen3-ASR-1.7B-ONNX-float16-delta
E:\ML\Audio\models\Qwen3-ASR-1.7B-ONNX-int8
```

> `*-delta` 是相对 baseline 的差分权重变体，`int8` 牺牲精度换速度/显存。横评时建议把同档（0.6B 或 1.7B）的三种放一组比，更有信息量。

---

## Quick start

> 假设当前工作目录在仓库根（含 `src/`），下面命令在 PowerShell / bash 都能直接复制。

### Smoke test（约 30 秒）

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- --count 30
```

### Standard（默认 500 条，几分钟）

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark
```

### Full test set

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- --full
```

---

## 场景 1：算法改动前后回归

工作流：先把当前 main 的结果存成 baseline，改算法后再跑一次并 `--compare`。

```bash
# Step 1: baseline（改算法前）
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- \
  --count 400 -o src/TestResults/baseline.json

# Step 2: 改完代码后再跑一次，diff 自动产出
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- \
  --count 400 --compare src/TestResults/baseline.json
```

`--compare` 干的事：

1. 加载 baseline JSON
2. 按 `audio_path` 匹配两边样本（跨 `--seed` / `--count` 也能取交集）
3. 打印 Δ 区块：`aggregate_cer / aggregate_rtf / exact_match_rate` 三项各自的 before → after + 改善/退步标记
4. 列出 top-5 回归样本（baseline 命中→当前出错 优先，其次 CER 上升最多者）
5. 把上述结果作为 `comparison` 字段写进本次输出的 JSON

> **跑多轮取均值**：GPU RTF 单轮抖动 ±5% 是常态。重要的优化建议跑 3~5 次再判断方向。

---

## 场景 2：跨模型横评

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- --count 400 --models "E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-float16,E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-float16-delta,E:\ML\Audio\models\Qwen3-ASR-0.6B-ONNX-int8"
```

- 每个模型顺序跑（不并行，避免抢 GPU 互相影响 RTF）
- 每个模型独立写一份 JSON：`src/TestResults/qwen3-asr-benchmark-<modelName>-<ts>.json`
- 末尾打印一张横向表：

  ```
  Model                                CER      Exact%   RTF      Wall(s)   Samples
  Qwen3-ASR-0.6B-ONNX-float16          0.70%    92.00%   0.105    26.8      400
  Qwen3-ASR-0.6B-ONNX-float16-delta    0.65%    93.00%   0.098    25.1      400
  Qwen3-ASR-0.6B-ONNX-int8             1.40%    87.50%   0.072    19.3      400
  ```

`--models` 和 `--compare` 互斥（baseline 是单模型概念）。

---

## 场景 3：AEC / DNR 前处理对 ASR 精度的影响（`--matrix`）

用来回答一个具体问题：**给采集链路接上 DFSMN-AEC / DPDFNet-DNR 之后，干净语音的识别精度是升还是降？**

AIShell-1 是干净的离线录音、benchmark 过程没有任何 TTS 回放，所以 AEC 运行时拿到的远端(render)参考**恒为全零**——这正好复刻"用户对着安静的助手说话、没有回声"这一最常见工况。`--matrix` 在**同一批样本**上顺序跑四种前处理配置，直接出对照：

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- --matrix --count 200
```

- 四路配置：`none` / `aec` / `dnr` / `both`，其中 `none` 为内置 baseline
- 每路独立写一份 JSON（文件名后缀 `-none/-aec/-dnr/-both`）
- 先打印一张横向表（复用 `--models` 的表格）：

  ```
  Model   CER       Exact%    RTF       Wall(s)   Samples
  none    0.70%     92.00%    0.105     26.8      200
  aec     1.55%     85.00%    0.108     41.2      200
  dnr     0.95%     90.00%    0.107     38.0      200
  both    1.70%     84.00%    0.110     52.6      200
  ```
  （上表数字为示意，非实测。）

- 然后对 `aec` / `dnr` / `both` 各打印一段 **vs none** 的对照：`aggregate_cer` 的 before→after + 改善/退步标记，以及 top-5 回归样本（哪几条 baseline 命中、加了前处理后反而错）

### 调优旋钮（默认对齐 `VoiceAssistantDemo`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--preproc <mode>` | `none` | 单跑某一种前处理（不走矩阵）：`none\|aec\|dnr\|both` |
| `--preproc-ep <ep>` | `cpu` | AEC/DNR 的 ONNX 执行后端。默认 CPU，避免与 ASR 抢 GPU（RNNoise 后端忽略此项） |
| `--dnr-backend <name>` | `dpdfnet` | DNR 实现：`dpdfnet`（DeepFilterNet ONNX）/ `rnnoise`（经典 48k 原生 GRU）。`rnnoise` 忽略 `--dnr-model`/`--atten-limit-db` |
| `--mask-floor-db <v\|off>` | `60` | DFSMN-AEC 神经掩码的抑制下限(dB)。`off`=不设限。**调低到 18 可验证"放宽掩码是否让精度回升"** |
| `--dnr-model <name>` | `dpdfnet8` | DPDFNet 档位（仅 `dpdfnet` 后端）：`baseline\|dpdfnet2\|dpdfnet4\|dpdfnet8` |
| `--atten-limit-db <v\|off>` | `6` | DPDFNet 单帧最大抑制量上限(dB，仅 `dpdfnet` 后端)。`off`=完全采用增强谱 |
| `--atten-mode <name>` | `complex` | `--atten-limit-db` 的施加方式（仅 `dpdfnet`）：`complex`（历史复数谱混合）/ `magfloor`（幅度域增益下限，保留 noisy 相位，对 ASR 更友好） |

> **实测对比（AIShell-1，Qwen3-ASR-0.6B，无回声）**：
> - 换后端：DPDFNet `dnr` 把 CER 从 1.58%→2.10%（+0.53pp）；`--dnr-backend rnnoise` 仅 +0.11pp 且快 ~30×。
> - **`AttenuationLimitDb` 是非单调的**：复数混合在 alpha≈0.5（即 `--atten-limit-db 6`，Demo 默认）处**最差**（相位干涉伪影），比 `off`/`3` 都差。
> - **`--atten-mode magfloor` 是结构性修复**：同样 6dB 上限下，幅度域增益下限把 CER 拉回基线（1.44% vs complex 的 1.91%，count=120），既保留 6dB 降噪深度又不伤 ASR。**留在 DPDFNet 时推荐 `--atten-mode magfloor`。**
> - **收益侧（`--add-noise`，白噪声）**：叠白噪声后 DNR 反而**全程帮倒忙**——SNR=10dB 时 none=3.18%，任何 DPDFNet 配置都更差（complex@6=4.97%、magfloor@6=3.93%、off 更糟）；SNR=5dB 同理（none=5.37% 最优）。即该 ASR 对白噪声的鲁棒性强过 DPDFNet 伪影。**但白噪声并非 DPDFNet 的训练分布（真实噪声为 babble/交通等），真实噪声下结论可能不同**——需用真实噪声样本（后续可加 `--noise-file`）复核。

典型的"定位掩码是否是元凶"实验——把 AEC 的掩码下限放宽再跑一次矩阵，看 `aec/both` 的 CER 是否回升：

```bash
dotnet run --project src/Tests/VoiceBox.ASR.Qwen3.Benchmark -- --matrix --count 200 --mask-floor-db 18
```

### 注意事项

- **前处理耗时单列**：`total_preproc_seconds` 字段独立记录，**不计入 `aggregate_rtf`**（RTF 仍只反映 ASR 本身）。
- **离线套用语义**：每条 utterance 整段喂入后用零做 flush 取回尾部，首尾各多约 40ms 静音，对离线 ASR 无害。每条之间复位 AEC/DNR 状态，互不串扰。
- **已知点**：AEC 的神经频谱掩码**无条件每帧运行**，即使 render 全零（无回声）也会套到近端语音上；默认 `MaskFloorDb=60` 约等于不设限。这正是 `--matrix` + `--mask-floor-db` 想要量化的对象。
- `--matrix` 与 `--models` / `--compare` / `--preproc` 互斥（它自带四路遍历与 vs-none 对照）。

---

## Output schema（截选关键字段）

```jsonc
{
  "timestamp": "2026-05-27T10:00:00",
  "dataset_csv": "...",
  "model_dir": "...",
  "execution_provider": "cuda",
  "sample_count": 200,
  "failed_count": 0,
  "aggregate_cer": 0.0070,
  "exact_match_rate": 0.92,
  "aggregate_rtf": 0.105,
  "warmup_seconds": 2.96,
  "wall_seconds": 26.83,
  "samples": [
    { "index": 1, "audio_path": "...", "reference": "...", "hypothesis": "...",
      "audio_seconds": 5.38, "inference_seconds": 0.56, "rtf": 0.103,
      "cer": 0.0, "substitutions": 0, "insertions": 0, "deletions": 0,
      "exact_match": true, "error": null }
    // ...
  ],
  "comparison": {  // 仅在 --compare 时出现
    "baseline_path": "...",
    "delta_aggregate_cer": -0.0008,    // 负数 = 精度改善
    "delta_aggregate_rtf": -0.012,     // 负数 = 速度改善
    "delta_exact_match_rate": +0.01,   // 正数 = exact-match 增多
    "regressed_samples": [ /* top-5 退步样本 */ ]
  }
}
```

---

## Tips for AI agents

- **退出码**：`0` 正常；`2` 参数 / 路径错；`130` Ctrl-C；当前不区分 ASR 内部失败（看 `failed_count`）。
- **输出路径**：未给 `--output` 时自动落到仓库 `src/TestResults/`，文件名带时间戳，不会覆盖历史结果，可放心反复跑。
- **JSON 是首选解析对象**，控制台 summary 是给人看的。需要数字直接 `jq '.aggregate_cer, .aggregate_rtf' file.json`。
- **首次 smoke 推荐 `--count 30`**，可在 ~30 秒拿到完整 schema 验证 CLI 是否正确。
- **CUDA 是默认 EP**，没有 GPU 加 `--ep cpu`（精度不变，速度慢一个数量级）。
- **不要尝试并行跑两个 benchmark 进程**，会抢 GPU 显存且让 RTF 互相干扰。横评请用 `--models`，runner 内部强制 `MaxConcurrentInferences=1`，已保证测量独立。
