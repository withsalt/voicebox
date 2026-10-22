# 小云唤醒词 (FSMN-KWS) 流式 ONNX 导出

把 ModelScope 语音唤醒模型 [iic/speech_charctc_kws_phone-xiaoyun](https://www.modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun)
导出为**带显式 cache 的流式 ONNX 模型**，为 C# 侧 (`VoiceBox.Wakeup.FsmnKws`) 提供低延迟、可自定义唤醒词的流式唤醒能力。

## 1. 项目介绍

### 1.1 原始模型

| 项 | 值 |
|---|---|
| 结构 | 4 层 cFSMN + CTC（论文 [Compact Feedforward Sequential Memory Networks for Small-footprint KWS](https://www.isca-speech.org/archive/interspeech_2018/chen18c_interspeech.html)） |
| 参数量 | ~750K，CPU 实时推理无压力 |
| 建模单元 | 中文 char 全集，共 2599 个 token（含 `<blank>`/`<unk>`） |
| 前端 | 16kHz → 80 维 Fbank（25ms 帧长 / 10ms 帧移，hamming 窗）→ LFR(m=5, n=3) → 400 维 / 30ms 帧 → CMVN |
| FSMN 层参数 | linear=250, proj=128, **lorder=10, rorder=2**, stride=1 |
| 训练目标词 | "小云小云"（basetrain 5000h+ ASR 数据 + 1 万条目标词微调） |

模型仓库中 `kwsr.*` 文件是闭源 kwsbp 引擎专用的 kaldi 二进制资源，**不参与导出**；
导出对象是 `funasr/` 目录下的 FunASR 格式 checkpoint
（`finetune_fsmn_4e_l10r2_250_128_fdim80_t2599_xiaoyun_xiaoyun.pt`）。

由于采用 char 全集建模，**唤醒词可任意配置**（任意中文词转 token 序列），不限于"小云小云"；
非目标词的实际唤醒率取决于与训练数据的匹配度，需自行测试。

### 1.2 为什么能做流式（以及本导出与 FunASR 官方实现的差异）

FSMN 是纯前馈网络，记忆块本质是 FIR 滤波器，没有 RNN/Attention 的无限状态。
每层只依赖过去 9 帧 + 未来 2 帧，因此流式推理只需为每层维护一个
`(lorder-1) + rorder = 11` 帧的投影缓存。

FunASR 官方 `FSMNBlock` 的 cache 实现对 rorder（右上下文）做的是**分块边界零填充近似**
（源码中留有 "maybe need to check" 注释），每个 chunk 的最后 2 帧输出会失真且随层间传播。
本导出改为**精确 lookahead 缓存**：

- 每层 cache 同时容纳左历史 (9 帧) 与右 lookahead (2 帧)，输出整体延迟 `4 层 × 2 帧 = 8 帧`（240ms）；
- 稳态输出与整段推理**逐位一致**（实测最大误差 3.6e-7），且与分块大小无关（逐帧喂和按 100 帧喂结果相同）；
- 代价：cache 清零重置后约 35 帧（~1s）预热期内输出不精确（无害，已验证静音/语音下均无误报）。

## 2. 目录结构

```
kws-xiaoyun-onnx-export/
├── README.md                      ← 本文档
├── 01_baseline_funasr.py          ← 步骤 1: FunASR 基线推理验证
├── 02_export_streaming_onnx.py    ← 步骤 2: 导出流式 ONNX + 资源
├── 03_verify_parity.py            ← 步骤 3: parity 验证 + dump C# 测试向量
├── model/
│   └── speech_charctc_kws_phone-xiaoyun/   ← ModelScope 模型仓库 (git clone)
└── outputs/                       ← 全部产物 (运行脚本后生成)
    ├── baseline/                  ← 步骤 1 的检测输出
    ├── onnx/                      ← 流式模型 + C# 运行时资源
    └── vectors/                   ← C# parity 测试向量
```

## 3. 导出步骤

### 3.1 环境准备

建议使用 conda 虚拟环境（Python 3.9 ~ 3.12），目前（2026-06-11）funasr依赖`editdistance`库，目前仅支持Python <= 3.12。

```bash
conda activate kwsexport
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install funasr onnx onnxruntime onnxscript soundfile
```

实际验证过的版本：`torch 2.8.0+cpu / funasr 1.3.9 / onnx 1.19.1 / onnxruntime 1.19.2`（env `kwsexport`）
与 `torch 2.12.0+cpu / funasr 1.3.9 / onnx 1.21.0 / onnxruntime 1.26.0`（env `xiaoyunkws312`）。
CPU 即可，整个导出过程不需要 GPU。

> Windows 控制台为 GBK 编码时，新版 torch 导出器打印 emoji 会抛
> `UnicodeEncodeError: 'gbk' codec ...`，运行脚本时加 `PYTHONUTF8=1`（或
> `set PYTHONIOENCODING=utf-8`）即可。

模型本体已在 `model/speech_charctc_kws_phone-xiaoyun/`；如需重新获取：

```bash
git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun.git model/speech_charctc_kws_phone-xiaoyun
```

### 3.2 步骤 1：基线验证（可选但建议）

```bash
python 01_baseline_funasr.py
```

用 FunASR `AutoModel` 加载本地模型目录，对自带样例
`example/kws_xiaoyunxiaoyun.wav` 做整段推理。预期输出：

```
RESULT: [{'key': 'kws_xiaoyunxiaoyun', 'text': 'detected 小云小云 0.9954625454358539'}]
```

该分数 (0.99546) 是后续所有验证的基准。

### 3.3 步骤 2：导出流式 ONNX

```bash
python 02_export_streaming_onnx.py
```

脚本内容：

1. 通过 `AutoModel` 加载模型权重，取出 FSMN encoder；
2. 用 `StreamingFsmnKws` wrapper 重写前向：每层投影帧缓存显式化为输入/输出张量（精确 lookahead 方案）；
3. **导出前自检**：随机输入下逐帧流式 vs 整段前向，稳态误差须 < 1e-5；
4. `torch.onnx.export`（opset 17，T 轴动态）；
5. **导出后自检**：onnxruntime 不规则分块 vs torch 流式，误差须 < 1e-4；
6. 拷贝 C# 运行时资源（tokens / lexicon / cmvn）并生成 meta.json 与部署 keywords.json。

预期输出：

```
steady-state max diff (torch stream vs full): 2.027e-06
exported: .../outputs/onnx/kws_streaming.onnx
onnxruntime vs torch max diff: 1.228e-05
keywords.json: .../outputs/onnx/keywords.json (keyword=小云小云, threshold1=0.5)
resources written to .../outputs/onnx
```

### 3.4 步骤 3：端到端 parity 验证 + 生成 C# 测试向量

```bash
python 03_verify_parity.py
```

验证四件事，并 dump C# 测试向量：

1. 真实音频上，分块（6 帧/次）ONNX 流式输出与 torch 整段推理稳态逐位一致；
2. 流式 CTC prefix beam search 解码器（C# 解码器的 Python 原型，含过期剪枝/重归一化/最大时长约束）能在样例音频上检出关键词；
3. 5 秒纯静音零误报;
4. dump `outputs/vectors/`（原始采样、LFR+CMVN 特征、流式后验）供 C# 测试逐级比对。

预期输出：

```
wav: 72462 samples (4.53s)
feats: (1, 151, 400)
real-wav steady-state parity (onnx stream vs torch full): 3.576e-07
keyword token ids: [1462, 976, 1462, 976]
DETECTED at output frame 88 (~2.64s) score=0.9955
silence false alarms: 0
ALL CHECKS PASSED
```

## 4. 导出产物

### 4.1 `outputs/onnx/` — C# 运行时资源（部署需要这一整套）

| 文件 | 说明 |
|---|---|
| `kws_streaming.onnx` | 流式 KWS 模型（~3MB, fp32） |
| `tokens.txt` | 2599 行 token 表，行号即 token id |
| `am.mvn` | kaldi 文本格式 CMVN（AddShift = 负均值，Rescale = 逆标准差，各 400 维） |
| `lexicon.txt` | 词典（关键词含英文单词时用于展开成 char token） |
| `kws_meta.json` | 特征参数 / 模型维度 / 延迟与预热帧数等元信息（文档用途） |
| `keywords.json` | 部署唤醒词配置（字段对齐 kws-train 导出；默认 `小云小云`，`threshold1=0.5`，命中分=各 token 峰值后验几何平均，scale (0,1]） |

### 4.2 ONNX 模型契约

```
输入:
  feats   float32 [1, T, 400]    # LFR+CMVN 后特征, T >= 1 任意
  cache0..cache3 float32 [1, 128, 11, 1]   # 流重置时全零
输出:
  probs   float32 [1, T, 2599]   # softmax 后验 (softmax 已在图内)
  new_cache0..new_cache3         # 回填下次调用
```

特征管线（与 FunASR WavFrontend 对齐，C# 侧需完全一致）：

```
float32 PCM [-1,1] ×32768 → kaldi fbank: 80 mel / 25ms 帧长 / 10ms 帧移 /
hamming 窗 / dither=0 / snip_edges=true / 预加重 0.97 / 去直流 /
功率谱 / 自然对数 / low_freq=20Hz / high_freq=8000Hz
→ LFR(m=5, n=3): 左侧补 2 帧首帧副本, 每 3 帧堆叠 5 帧 → 400 维 / 30ms
→ CMVN: (x + AddShift) * Rescale
```

时序约定：

- 输出帧 j 对应输入帧 j-8（固定 240ms lookahead 延迟）；
- cache 清零后前 ~35 个输出帧为预热期（数值与整段推理不一致，但无误报风险）；
- 解码用 CTC prefix beam search（score_beam=3, path_beam=20, 后验>0.05 过滤，
  命中分 = √(关键词各 token 峰值后验之积)）。

### 4.3 `outputs/vectors/` — C# parity 测试向量

| 文件 | 内容 |
|---|---|
| `input_samples.bin` | 样例 wav 原始采样（float32 LE, 16kHz mono） |
| `feats.bin` | LFR+CMVN 后特征 `[151, 400]`（float32 LE 行优先） |
| `probs_stream.bin` | ONNX 流式后验 `[151, 2599]` |
| `vectors_meta.json` | 形状 / 关键词 token id / 期望检出帧与分数 |

## 5. C# 侧接入

- 服务实现：`src/VoiceBox.Wakeup.FsmnKws/`（实现 `IWakeupService`，特征提取 / 流式推理 / 解码全部纯 C#）
- parity 验证依赖上面的 outputs 产物，缺失时自动跳过
- Demo：`src/Demos/FsmnKwsDemo/`（wav 推流 + 麦克风两种模式，模型目录默认指向本目录 `outputs/onnx`，可用环境变量 `FSMN_KWS_MODEL_DIR` 覆盖）

样例音频实测（与 Python 基线一致）：

```
>>> [#1] 命中: keyword=小云小云 score=100% threshold=50% t=1890ms duration=540ms
```

## 6. 常见问题

- **`AttributeError: 'FsmnKWS' object has no attribute 'writer'`**：
  FunASR 1.3.9 的 bug，`model.generate(...)` 必须传 `output_dir`（检测其实已成功，只是写结果时崩）。
- **parity 验证时特征对不上**：推理时 `WavFrontend` 默认 `dither=1.0`（随机加噪，非确定性）。
  做数值比对必须显式 `dither=0`，本仓库脚本均已处理。
- **C# 进程 0xC0000005 崩溃**：可执行/测试项目只引用了 `Microsoft.ML.OnnxRuntime.Managed`
  时没有原生库，P/Invoke 会加载到 `C:\Windows\System32\onnxruntime.dll`（系统旧版）导致访问冲突。
  必须额外引用带原生库的 `Microsoft.ML.OnnxRuntime`（CPU）包。
- **更换/微调模型后重新导出**：微调产出新 `.pt` 后替换 `funasr/finetune_*.pt`
  （或修改 `configuration.json` 的 `init_param`），重跑步骤 2、3 即可；
  网络结构参数（层数/lorder/rorder/proj_dim）若变化，需同步修改脚本顶部常量与 C# 侧 `StreamingFsmnModel` 常量。
