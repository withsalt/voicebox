# VoxCPM-0.5B → ONNX 导出

把 **VoxCPM-0.5B**(OpenBMB tokenizer-free TTS)导出为 C#（`VoiceBox.TTS.VoxCPM`）可直接消费的 ONNX 部署包，并对每个组件做数值校验（fp32 下端到端 SNR ≥ 80dB）。

***

## 概述

模型是"自回归 LM + 局部扩散 + Audio VAE"混合管线（16 kHz，每 AR 步产出 80 ms 音频）：

```
文本 ─tokenize(+中文拆字)→ prefill (base_lm 24L + FSQ + residual_lm 6L)
   循环: cfm_decode (10步Euler流匹配, CFG=2.0) → 1 patch latent
         → feat_encoder → lm 单步 (KV cache) → stop 头判停
   收尾/流式: audio_vae_decoder → 波形
```

**双层循环与停止判断留在宿主（C#）**；ONNX 只承载 5 个纯张量子图。CFM 噪声、
文本 embedding 查表都由宿主喂入 → 随机性可控，跨 torch/ORT/C# 可复现。

| 工件 | 内容 | 精度 |
| --- | --- | --- |
| `lm.onnx` | 统一 prefill/decode 图（base+residual LM、FSQ、dit/stop 头、KV cache 增长） | 随 `--dtype` |
| `feat_encoder.onnx` | latent patch → LM 嵌入（prompt 一次 + 每步 T=1） | fp32（fp16 包为 fp16） |
| `cfm_decode.onnx` | 整段 10 步 Euler 流匹配（CFG batch=2 + cfg-zero-star + sway），噪声/cfg 为输入 | fp32（fp16 包为 fp16） |
| `audio_vae_decoder.onnx` | latents → 16 kHz 波形（流式 3-patch 滑窗 + 收尾整段） | **恒 fp32** |
| `audio_vae_encoder.onnx` | 波形 → latents（音色克隆 prompt，一次性） | **恒 fp32** |
| `embeddings/text_embedding.bin` | [73448,1024] fp32 文本嵌入查表（宿主 Gather） | — |
| `embeddings/multichar_zh.json` | 多字中文 token 拆单字映射（19697 项，tokenizer 语义对齐） | — |
| `config.json` / `metadata.json` / `tokenizer/` | C# manifest + 生成默认值 + HF tokenizer | — |

`_engine.py` 是纯 numpy + ONNXRuntime 的**参考引擎**，等价于 C# 流水线，既是 e2e parity 的参照，也是 C# 移植的可执行规格。

***

## 环境

conda 环境 `voxcpm`（Python 3.13，torch 2.12.1+cu132）。直接用环境的 `python.exe` 运行（`conda run` 偶发异常）：

```bash
E:/.packages/conda/envs/voxcpm/python.exe export.py --dtype fp32
```

缺包按 `requirement.txt` 安装。本机三坑（详见 WORKLOG.md）：cuDNN 子库不匹配 → 全程 CPU 导出；torchaudio 2.11 load 需 torchcodec → 音频一律 soundfile 解码；`conda run` 不稳 → 直呼 python.exe。

源模型目录需含：`config.json`、`pytorch_model.bin`、`audiovae.pth`、`tokenizer.json`。本仓库已带 `VoxCPM-0.5B/` 与上游源码 `VoxCPM/`。

***

## 用法

一条命令导出整包（embeddings → lm/feat_encoder → cfm → vae → package → parity+golden）：

```bash
# fp32（与 PyTorch 噪声对拍 SNR≥80dB，推荐用于校验 / golden 生成）
python export.py --model-dir VoxCPM-0.5B --dtype fp32

# fp16（原生半精 trace，lm/cfm/feat_encoder 半精，~1.7GB）
python export.py --dtype fp16

# int4（lm 权重-only MatMulNBits，lm 1811→303MB，整包 ~1.4GB）
python export.py --dtype int8
```

也可单独跑某一步（共用同一组参数）：

```bash
python export_lm.py --dtype fp32          # 仅 lm + feat_encoder（含自查）
python export_cfm.py --dtype fp32         # 仅 cfm_decode
python export_audio_vae.py                # 仅 VAE encoder/decoder
python parity_e2e.py --dtype fp32         # 仅端到端 parity + golden
```

### 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model-dir` | `./VoxCPM-0.5B` | 原始模型目录 |
| `--out-dir` | `./out` | 部署包 → `<out-dir>/voxcpm-0.5b-onnx-<dtype>/`，golden → `<out-dir>/golden/` |
| `--dtype` | `fp32` | `fp32` / `fp16` / `int8` |
| `--opset` | `20` | ONNX opset |
| `--device` | `cpu` | trace 设备（本机 cuDNN 不匹配，保持 cpu） |
| `--skip-parity` | 关 | 跳过末尾 e2e 校验（仅 `export.py`） |

***

## 校验（parity）

- **逐图自查**（每个 export_*.py）：改写 forward vs 原始模块、ORT vs torch，全部 SNR > 106dB。
- **e2e**（`parity_e2e.py`，fp32 严格模式）：向 torch 的 `UnifiedCFM.forward` 注入与 ONNX 引擎相同的
  numpy 噪声序列，比较完整波形。en 80.8dB / zh 100.1dB / clone 86.7dB / 流式 80.7dB，步数逐一致。
- **fp16/int8**：冒烟模式（有限值/非静音/长度合理）。扩散轨迹在低精度下天然分叉，
  样本级对拍无意义；听感验证 `out/baseline/int4_*.wav`。
- **golden**（fp32 时采集，C# 测试消费）：`out/golden/` 含每 case 的
  `text_ids/latents/wav/noise` npy、`wav_stream_en.npy`、`prompt_16k.wav`、`prompt_feat.npy`、`golden_meta.json`。
  ⚠️ AR 环路混沌（FSQ round 放大微差）：C# 对拍必须复用 golden 的噪声与 prompt 特征，见 WORKLOG。

## C# 侧约定

`VoxCPMTTSOptions.ModelPath` 指向部署包目录。宿主循环规格 = `_engine.py::VoxCpmOnnxEngine.generate`：

1. tokenize（不加 BOS）→ multichar_zh 拆分 → 追加 `audio_start=101`；
2. prefill：text_embedding 查表 + prompt latent（可选 VAE encode，左填充到 1280 倍数）→ `lm.onnx`(P=0)；
3. 每步：`cfm_decode`(noise 宿主 RNG) → `feat_encoder`(T=1) → stop argmax（`i>min_len` 才生效）→ `lm.onnx`(S=1)；
   stop 用**本步开头**的 stop_logits（即上一次 lm 调用的输出）；
4. 流式：每步取最近 3 patch → `audio_vae_decoder` → 取尾 1280 样本；
   非流式：全部 patch（含 prompt 尾 2 patch 上下文，解码后丢弃其样本）一次解码。
