# VoiceBox.DNR.Benchmark

多个 DNR(降噪 / 语音增强)后端的**降噪效果 + 性能**横评工具。用参考类信号域指标直接量化各后端在受控噪声下把语音"洗干净"的能力,并采集 RTF / 逐帧延迟 / 算法延迟等性能参数。结果写成 JSON 落到 `src/TestResults/`,末尾打印跨后端对照表。

评测思路一句话:**干净语音 + 噪声@SNR → 各 DNR → 时间对齐 → 对比干净参考算指标**。同一 (utterance, SNR) 的含噪信号对每个后端**字节一致**(路径+SNR 稳定哈希播种),对照公平。

---

## 覆盖的后端

| token | 实现 | 采样率 | 备注 |
|---|---|---|---|
| `deepfilternet` | DeepFilterNet3-LL ONNX 流式 | 48k native | 16k 输入库内重采样 |
| `dpdfnet` / `dpdfnet:baseline\|2\|4\|8` / `dpdfnet-all` | DPDFNet(DeepFilterNet 系)ONNX | 16k native | 默认 dpdfnet8;`-all` 展开四档 |
| `hush` | Hush(DFN3-SE @16k)ONNX 流式 | 16k native | |
| `rnnoise` | 经典 48k 原生 GRU | 48k native | native,忽略 `--ep`,极快 |
| `webrtc` | WebRTC APM 噪声抑制(EC 关 / NS 开 / HPF 开 / AGC 关) | 16k | native,传统信号处理对照 |

---

## What it measures

**降噪效果(质量,越高越好)**

- **SI-SDRi(dB)** — 主指标。`SI-SDR(clean, enhanced) − SI-SDR(clean, noisy)`,即增强信号相对含噪输入向干净参考靠近了多少。SI-SDR 尺度不变。
- **分段 SNRi(dB)** — 帧级(25ms/10ms)SNR 提升,仅统计语音活跃帧,贴近主观降噪感。评分前对 estimate 做全局最小二乘增益对齐,只度量失真不受音量差影响。
- **干净语音保真(dB SI-SDR)** — "代价"侧:干净语音 → DNR → 对比干净参考。无噪可去时模型若损伤语音则得分低。呼应生产中"前处理是否在干净语音上掉点"的关注。

**性能(越低越好)**

- **RTF** = 处理耗时 / 音频时长(逐帧流式驱动);及其倒数 `xRealtime`。
- **逐帧延迟 p50/p90/p99(ms)** — 每个实时帧/块 `Denoise`/`Process` 调用耗时分位(不含 flush)。
- **算法延迟(ms)** — 实测的固定流水线延迟(每后端预热时以互相关估出的滞后换算),反映端到端时延下限。
- **warmup / wall** — 首帧(含模型加载)耗时、整批墙钟。

---

## Defaults

`Program.cs` 顶部常量 / `ParsedArgs` 默认值;机器不同用 CLI 覆盖。

| 项 | 默认值 |
|---|---|
| `--clean` | AIShell-1 测试 CSV(与 ASR benchmark 同一常量) |
| `--noise` | 无 → 合成白噪声 |
| `--snr` | `5,10,15` |
| `--count` | `200` |
| `--seed` | `42` |
| `--backends` | `deepfilternet,dpdfnet,hush,rnnoise,webrtc` |
| `--ep` | `cpu` |
| `--ns-level` | `high`(仅 webrtc) |

---

## Quick start

> 假设当前工作目录在仓库根(含 `src/`)。

### Smoke(快、native、免 GPU)

```bash
dotnet run --project src/Tests/VoiceBox.DNR.Benchmark -- --count 10 --snr 10 --backends rnnoise
```

### 真实噪声全后端横评

```bash
dotnet run --project src/Tests/VoiceBox.DNR.Benchmark -- \
  --count 100 --snr 5,10,15 \
  --noise E:\ML\Audio\noise\DEMAND \
  --backends deepfilternet,dpdfnet-all,hush,rnnoise,webrtc
```

末尾横评表(数字为示意):

```
Backend        SISDRi    segSNRi   CleanFid   RTF       p90 ms    algo ms   Wall(s)   Pairs
------------------------------------------------------------------------------------------
deepfilternet  8.12      6.40      24.30      0.180     0.42      20.0      33.1      300
dpdfnet-8      9.05      7.11      21.80      0.240     0.55      12.5      41.2      300
hush           8.60      6.85      22.40      0.150     0.35      10.0      28.7      300
rnnoise        4.20      3.10      27.60      0.006     0.02      10.0      2.1       300
webrtc-ns      3.10      2.40      30.10      0.010     0.03      10.0      2.6       300
```

### 导出音频试听

```bash
dotnet run --project src/Tests/VoiceBox.DNR.Benchmark -- \
  --count 3 --snr 5 --backends dpdfnet,rnnoise --dump-dir src/TestResults/dnr-audition
```

每条导出 `<backend>__<stem>__{clean,noisy_snrX,enh_snrX}.wav`,直接耳朵核对降噪是否生效。

---

## Output schema(关键字段)

```jsonc
{
  "backend": "dpdfnet-8",
  "timestamp": "2026-07-01T10:00:00",
  "clean_path": "...", "noise_path": "...", "noise_type": "real",
  "execution_provider": "cpu",
  "snr_levels": [5, 10, 15],
  "sample_count": 100, "failed_count": 0, "pair_count": 300,
  "mean_si_sdr_i": 9.05,          // 主降噪指标, dB, 越高越好
  "mean_seg_snr_i": 7.11,         // dB
  "mean_clean_fidelity": 21.80,   // dB SI-SDR, 干净语音保真
  "per_snr": [
    { "snr_db": 5, "count": 100, "mean_in_si_sdr": 5.0, "mean_out_si_sdr": 13.9,
      "mean_si_sdr_i": 8.9, "mean_seg_snr_i": 6.9 }
    // ...
  ],
  "aggregate_rtf": 0.240, "throughput_x_realtime": 4.2,
  "latency_p50_ms": 0.30, "latency_p90_ms": 0.55, "latency_p99_ms": 0.90,
  "algorithmic_latency_ms": 12.5,
  "warmup_seconds": 0.8, "wall_seconds": 41.2,
  "total_audio_seconds": 512.0, "total_proc_seconds": 122.9,
  "clean_cost": [ { "audio_path": "...", "si_sdr": 21.8 } ],
  "samples": [
    { "index": 1, "audio_path": "...", "snr_db": 5,
      "in_si_sdr": 5.0, "out_si_sdr": 13.9, "si_sdr_i": 8.9, "seg_snr_i": 6.9,
      "audio_seconds": 5.1, "proc_seconds": 1.2, "rtf": 0.235, "error": null }
    // ...
  ]
}
```

---

## Caveats(读数前必看)

- **统一 16 kHz mono 输入**(部署管线速率)。48k-native 模型(DeepFilterNet / RNNoise)库内做 16k↔48k 重采样——这是它们在本管线的真实代价,已计入 RTF 与质量,不额外扣分。
- **SI-SDR 时移敏感**:评分前用互相关把 enhanced 对齐到 clean(每后端一次固定滞后)。延迟单列为"算法延迟",不与降噪质量混淆。
- **SNR 以整段功率计**(含首尾静音),各后端一致,只影响绝对 SNR 解读不影响对照。
- **白噪声对神经 SE 属分布外**(交通/babble 等真实噪声才是训练分布)。**要得到有意义的神经模型质量数,务必用 `--noise` 指向真实噪声集**(DEMAND/DNS);白噪声仅作无数据时的自包含回退,可能低估神经模型。
- **RTF 单轮抖动**:CPU/GPU 单轮 ±若干 % 常见;重要结论多跑几轮。各后端顺序跑(不并行),避免互抢资源污染 RTF。
- **WebRTC-NS** 走 EC 关 / NS=High / HPF 开 / AGC 关的纯噪声抑制路径;`--ns-level` 调强度。

---

## Tips for AI agents

- **JSON 是首选解析对象**,控制台 summary 给人看。取数直接 `jq '.mean_si_sdr_i, .aggregate_rtf' file.json`。
- **退出码**:`0` 正常;`2` 参数 / 路径错;`130` Ctrl-C。后端内部失败不改退出码,看横评表的 `ERROR:` 行与 `failed_count`。
- **首次 smoke 用 `--count 10 --backends rnnoise`**,~秒级拿到完整 schema 验证 CLI。
- **默认 EP 是 cpu**;有 GPU 且要测 ONNX 后端加速用 `--ep cuda`(需 CUDA 运行时)。RNNoise/WebRTC 为 native,与 `--ep` 无关。
- **别并行跑两个进程**,会互抢 CPU/GPU 让 RTF 失真。
