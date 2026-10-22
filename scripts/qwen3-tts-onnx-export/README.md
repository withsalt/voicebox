# Qwen3-TTS → ONNX 导出

把 **Qwen3-TTS-12Hz-CustomVoice**（0.6B / 1.7B）导出为 C#（`VoiceBox.TTS.Qwen3`）可直接消费的 ONNX 部署包，并对每个组件做数值校验（贪心 codes 与 PyTorch **逐 token 字节一致**）。

***

## 概述

模型是两级自回归 + 神经 codec：

```
文本 ─tokenize→ 拼 prompt 嵌入 ─→ talker (主自回归, 每帧出第0组 codec token)
                                    └→ code_predictor (子talker, 每帧再自回归 15 步补满 16 组)
                                          └→ codec_decoder (16 组 codec → 24kHz 波形)
```

**只有 transformer / codec 进 ONNX；embedding、prompt 构造、talker 采样留在 C#**（轻量、易调、便于复用分词器）。导出产出 4 类工件，由 C# 流水线编排：

| 工件                                                            | 内容                                                                                               | 精度          |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------- |
| `talker.onnx`                                                 | 统一 prefill/decode 的主自回归图（KV-cache 增长）                                                            | 随 `--dtype` |
| `code_predictor.onnx`                                         | **融合的整帧子talker图**：单次 Run 内展开 15 步自回归 + 图内查表 + 图内采样，产出整帧 15 个 fine code；作为默认采样路径与旧运行时兼容图          | 随 `--dtype` |
| `code_predictor_sample.onnx` / `code_predictor_greedy.onnx`   | sample-only / greedy-only 专用整帧子talker图；C# 在 greedy subtalker 路径优先使用 `code_predictor_greedy.onnx` | 随 `--dtype` |
| `codec_decoder.onnx`                                          | codec → 24kHz 波形（chunked\_decode）                                                                | **恒 fp32**  |
| `embeddings/*.bin + config.json`、`tokenizer/`、`metadata.json` | 预投影文本嵌入/codec 嵌入 + 全量模型/生成配置 + Qwen BBPE 分词器 + 包元信息                                              | —           |

`_engine.py` 是纯 numpy + ONNXRuntime 的**参考引擎**，等价于 C# 流水线，既是 e2e parity 的参照，也是 C# 移植的可执行规格。

***

## 环境

conda 环境 `qwen3tts`（Python 3.13，torch 2.8+cu129）。直接用环境的 `python.exe` 运行（`conda run` 偶发异常）：

需要的包（若缺）： 

```bash

#如果使用 python 3.12 ~ 3.13  + cuda 12.9
pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu129 

#如果使用 python >= 3.14 + cuda 13.x
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

pip install numpy transformers>=4.57.3 accelerate librosa soundfile onnx onnxruntime onnxconverter-common onnxscript jmespath sox einops qwen_tts

```

> 说明：`transformers` 懒加载 boto3 → 需 `jmespath`；25Hz 分词器 import 需 `sox`+`einops`；`torchaudio` 2.11 与 torch 2.8 ABI 不兼容，须钉 2.8.0+cu129。`numpy` 是脚本硬依赖（若报 `No module named 'numpy'` 直接装）。

源模型目录需含：`config.json`、`model.safetensors`、`vocab.json`、`merges.txt`、`speech_tokenizer/config.json`。本仓库已带 `Qwen3-TTS-12Hz-0.6B-CustomVoice/` 与 `…-1.7B-CustomVoice/`。

***

## 用法

一条命令导出整包（codec → embeddings → talker → code\_predictor → package → parity）：

```bash
# 0.6B，fp32（与 PyTorch 贪心逐 token 字节一致，推荐用于校验 / 高保真部署）
python export.py --model-dir Qwen3-TTS-12Hz-0.6B-CustomVoice --dtype fp32

# 0.6B，bf16（默认；权重/KV 半精，体积减半，数值稳定）
python export.py --model-dir Qwen3-TTS-12Hz-0.6B-CustomVoice

# 1.7B，bf16
python export.py --model-dir Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bf16
```

也可单独跑某一步（共用同一组参数）：

```bash
python export_talker.py --dtype fp32          # 仅导出 talker（含自查）
python export_code_predictor.py --dtype fp32  # 仅导出融合 code_predictor
python parity_e2e.py --dtype fp32             # 仅跑端到端 parity
```

### 参数

| 参数              | 默认                                  | 说明                                                                                    |
| --------------- | ----------------------------------- | ------------------------------------------------------------------------------------- |
| `--model-dir`   | `./Qwen3-TTS-12Hz-0.6B-CustomVoice` | 原始模型目录                                                                                |
| `--out-dir`     | `./out`                             | 导出基目录。部署包 → `<out-dir>/qwen3-tts-<size>-onnx-<dtype>/`，golden → `<out-dir>/golden/<size>/` |
| `--dtype`       | `fp32`                              | `fp32` / `bf16` / `fp16` / `int8` / `int4`（talker & code\_predictor 精度；codec\_decoder 恒 fp32）  |
| `--opset`       | `20`                                | ONNX opset                                                                            |
| `--device`      | `cuda:0`                            | trace / 半精导出设备                                                                        |
| `--skip-parity` | 关                                   | 跳过末尾的 e2e PyTorch 校验（仅 `export.py`）                                                   |

### 环境变量

| 变量                       | 作用                                                         |
| ------------------------ | ---------------------------------------------------------- |
| `EXPORT_STATIC_TALKER=1` | 额外导出实验性 `talker_static.onnx`（定长 KV 静态图，见“其他说明”）。暂不可用，默认不导出 |

输出包布局：

```
out/qwen3-tts-0.6b-onnx-fp32/
├─ talker.onnx                     主自回归 (KV-cache)
├─ code_predictor.onnx             融合子talker (整帧一次 Run, legacy/default sample path)
├─ code_predictor_sample.onnx      sample-only 子talker
├─ code_predictor_greedy.onnx      greedy-only 子talker
├─ codec_decoder.onnx              codec → 24kHz (恒 fp32)
├─ embeddings/
│  ├─ config.json                  全量模型 + generation 配置 (C# ModelConfig/EmbeddingStore 读取)
│  ├─ text_projected_embedding.bin [Vtxt,H] = text_embedding + text_projection(fp32, 预计算)
│  ├─ codec_embedding.bin          [Vc,H]
│  └─ cp_codec_embedding.bin       [15,Vcp,H]
├─ tokenizer/{vocab.json,merges.txt,tokenizer_config.json}
└─ metadata.json                   dtype/opset/采样率/图名映射
```

C# 侧把 `Qwen3TTSOptions.ModelPath` 指向此目录即可。

***

## 校验（parity）

`export.py` 末尾对 **fp32** 跑端到端 parity（捕获 PyTorch `generate_custom_voice(do_sample=False, subtalker_dosample=False)` 的 golden）：

- `prompt_construction`：C# prompt 构造 vs 模型自身 ≈ **141 dB**
- `e2e_codes_match`：引擎贪心 codes 与 golden **逐 token 字节一致（0/448 失配）**
- `codec / e2e_wav`：解码波形 SNR ≈ **113.5 dB**

闸门：`SNR ≥ 40 dB` 且 `cos ≥ 0.999`，否则 `sys.exit(1)`。各 `export_*.py` 还在导出时自查（manual-layers vs 原始、KV-cache 一致性、ORT vs torch）。bf16/fp16/int8 的贪心会按预期 argmax-flip，仅报告不闸门。golden 工件存到 `out/golden/<size>/`，供回归验证复用。

***

## 精度说明

| dtype             | 体积    | 说明                                                      |
| ----------------- | ----- | ------------------------------------------------------- |
| **fp32**          | 最大    | 与 PyTorch 贪心**逐 token 字节一致**；高保真 / 校验首选，C# demo 默认      |
| **bf16**（默认）      | \~1/2 | 该模型 bf16 训练；与 fp32 同指数范围，CUDA 上数值稳定、逐帧贴合 fp32。省显存       |
| **fp16**（不推荐）     | \~1/2 | 指数范围过窄 → CUDA 上残差流漂移、陷入重复、不发 eos。仅 CPU 勉强可用，**请改 bf16** |
| **int8**（仅权重动态量化） | \~1/4 | 贪心输出较 fp32 偏离大（权重量化精度损失），音质下降，不建议用于高保真                  |
| **int4**（权重-only block-wise） | \~1/6 | 体积最小、CPU/DirectML 提速；贪心整段漂移（数据无关 RTN），**低保真/省资源**档，详见下方专节 |

> **速度：精度不是速度杠杆（GPU 上）。** 本机（RTX 5070 Ti）实测 bf16 ≈ fp32 每帧耗时（自回归是 kernel-launch 受限，非显存带宽受限）。所以 fp16/bf16 按\*\*保真度（fp32 字节精确）/ 显存（bf16 减半）\*\*取舍，而非速度。早期“fp32 RTF 0.80 vs bf16 0.91”的说法未能复现，已纠正。**例外是 int4**：在 CPU/DirectML（带宽受限）路径上 int4 会明显提速，见下。

### int4（MatMulNBits 权重量化）

- **量化范围**：talker 的全部 197 个权重矩阵 + code\_predictor 5 层 transformer + cp 的 head/codec-embedding 查表，全部转 `com.microsoft.MatMulNBits`（block-wise int4，`block_size=32`，非对称 RTN，数据无关）。注意力的 q·kᵀ / w·v（激活×激活）与 codec\_decoder（Conv）**保持浮点**。文本/codec 嵌入 `.bin`（最大 622MB 文本表）不是 ONNX、不受影响。
- **实现**：和 int8 一样从 fp32 图后处理（不走 native trace）。cp 是「15 步自回归整帧展开」图——5 层权重通过 `Identity` 在 15 步间共享，**必须先把 Identity 折叠回 initializer 再量化**（否则量化器只能量化每权重的那一个直接引用、fp32 原件还被其余 Identity 引用无法释放，结果反而更大：568MB→407MB）。折叠后每权重只量化一次：**568MB→88MB**。详见 `_common._fold_identity_weights` / `_hoist_const_matmul_weights`。
- **体积（0.6B 实测）**：talker 1775→271MB（6.5×）、每个 cp 568→88MB（6.5×）、整包 **4.5GB→1.7GB**（余量是 761MB fp32 `.bin` 嵌入 + 435MB fp32 codec，这两块不量化）。
- **速度**：CPU/DirectML 是显存带宽受限的，int4 实测 talker 单步 decode **\~1.9× 提速**；CUDA 是 kernel-launch 受限的，基本无提速、甚至因 dequant 略慢。Demo 默认引 `Microsoft.ML.OnnxRuntime.DirectML`，故 int4 在默认部署路径上「又小又快」。
- **精度（重要）**：数据无关 RTN 下贪心**不再字节一致**——e2e parity 实测整段 codes 全部漂移、并可能提前触发 eos（28 帧→22 帧）。属低保真档，适合「省体积/CPU 提速、可接受音色与 fp32 不同」的场景。要把质量拉回，需改用 **GPTQ/AWQ + 标定数据**（喂真实 talker 激活，工作量更大）。parity 对 int4 只报告不闸门（与 bf16/fp16/int8 一致）；被闸门的 prompt 构造 / codec 解码是 dtype 无关的，仍 ≥140dB / ≥112dB 通过。
- **运行时前提**：C# 侧的 ORT ExecutionProvider 必须支持 `MatMulNBits`（CPU / CUDA / DirectML 均支持；当前引用的 DirectML 1.24.4 可用）。

***

## 设计要点

- **预投影文本嵌入**：导出时用 PyTorch fp32 预计算
  `text_projection(text_embedding)` 为 `[Vtxt,H]` 查表，C# 运行时不再为每个文本 token
  执行两层 MLP。prompt parity 仍按 e2e gate 校验，旧包 schema 仍可由 C# fallback 读取。
- **融合 code\_predictor**：子talker 的 15 步自回归整体塞进单图。
  - 入：`past_hidden[1,1,H]`、`emb0[1,1,H]`（=codec(group0)，C# 查好）、`do_sample`、`temperature`、`randoms[15]`；出：`fine_codes[15]`、`next_sum[1,1,H]`（=codec(group0)+Σcp\_emb，C# 只需加 tts\_pad）。
  - 图内查表用 **one-hot @ 嵌入矩阵乘**（而非 argmax→Gather 的整数索引），避免 ORT 把索引算子钉到 CPU 产生逐步 memcpy。
  - 图内采样 = `temperature → top-k → softmax → 逆 CDF`（贪心走 argmax，`Where(do_sample,…)` 一图两用）；`top_k` 在导出时从 `generation_config.subtalker_top_k` 烘焙（静态形状），逆 CDF 钳 `u≤cdf[-1]` 防全零 one-hot。**贪心仍字节一致。**
  - 额外导出 sample-only / greedy-only 专用图。ORT 1.26 下默认采样路径仍优先使用 legacy mixed 图（实测更快），`subtalker_do_sample=false` 时使用 greedy-only 图。
- **talker**：统一 prefill(P=0)/decode(P>0) 单图，KV 随帧增长（动态轴）。
- **legacy tracer**（`torch.onnx.export(dynamo=False)`）规避坑：HF masking 用 `torch.vmap`（不可 trace）→ 自建加性 mask；`.view(*shape[:-1],…)` 会烤死序列长 → 用 `reshape(B,-1,…)`；CausalConv 动态 pad（codec）→ monkeypatch stride-1 返 0；手动循环 `model.layers` 跳过 HF 的 `cache_position=arange` 烤死。
- **mrope → 标准 1D rope**：CustomVoice 无视觉模态，三 mrope 位置行相同，交错合并 == 行0。注意力分数图内升 fp32（fp16-only，避免 CUDA Tensor Core 溢出；bf16/fp32 走原生 matmul）；mask 恒 fp32（CumSum 无 bf16 CUDA kernel）。
- **>2GB 权重合并**：torch 把 >2GB 模型（如 1.7B talker）散权重自动外置为多文件，`_common.export_onnx` 会合并成单 `<name>.onnx.data`（<2GB 模型不受影响、保持字节一致；`size_threshold=1024` 保标量内联，避开 onnx 标量合并 bug）。

***

## 其他说明

- **实验性** **`talker_static.onnx`（默认不导出）**：定长 `[L,1,kv,2048,D]` KV + `cache_position` ScatterND 写入 + 全长 mask 的纯静态 decode 图，**已验证与动态 talker 字节级等价（100.8 dB）**，为 CUDA-Graph 加速预留。但 C# 运行时**未接入**：ORT C# 无法 host→device 写设备 KV 缓冲（`GetTensorMutableDataAsSpan` 在 CUDA 张量上 AccessViolation），无法 seed/zero 定长缓冲；故只保留导出代码 + 校验，`EXPORT_STATIC_TALKER=1` 可按需生成。
- **运行时性能**不在本目录范围（属 C# 侧）：当前已落地 融合cp + talker IO-binding + codec/生成线程重叠 + 小首块调度，首包 \~250ms、RTF \~0.63、贪心字节一致。详见 C# 记忆 `qwen3-tts-csharp-inference`。
- **故障排查**：`No module named 'numpy'` → 装 numpy；`SoX could not be found` / `flash-attn is not installed` 警告无害（25Hz 分词器 / 注意力回退，可忽略）；fp32 包约 4GB、bf16 约 2.9GB（`text_embedding.bin` 1.24GB 是最大单文件）。
- 不修改 `Qwen3-TTS/` 源码树；所有导出逻辑在本目录的 `export_*.py` / `_common.py` / `_engine.py`。

C# 推理见 `src/VoiceBox.TTS.Qwen3`，demo 见 `src/Demos/Qwen3TtsDemo`（`--bench "<文本>"` 无头测首包/RTF）。
