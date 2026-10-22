# DeepFilterNet ONNX 导出 / 流式化 / 对拍工具

本目录用于**构建** `VoiceBox.DNR.DeepFilterNet`（C# DF3-LL 逐帧低延迟降噪库）所需的
有状态流式 ONNX 模型，并提供与原始模型**逐级对拍**的 Python 参考实现。

## 这个项目解决什么问题

DeepFilterNet 官方发布的是「**离线** ONNX」（`enc.onnx` / `erb_dec.onnx` / `df_dec.onnx`）：
模型内部 GRU 的初始隐状态被**固化为 0、不对外暴露 I/O**，整段序列一次性推理。Rust 参考实现
靠 `tract` 的脉冲化（PulsedModel）才能逐帧流式，而 **ONNX Runtime 没有该能力**——直接逐帧喂
会导致 GRU 状态每帧清零、输出错乱。

本目录的脚本把离线 ONNX 转换为「**有状态流式** ONNX」`combined_streaming.onnx`：把编码器/ERB
解码器/DF 解码器三处共 6 个 GRU 层的隐状态**外置为模型输入/输出**（`h0/h1`、`erb_h0/erb_h1`、
`df_h0/df_h1`），并把时间轴在进 GRU 前切到末帧，从而在 ONNX Runtime 下实现**真·逐帧（10ms）
低延迟**推理。整个转换仅依赖 `onnx + numpy`(+可选 `onnxsim`)，**无需 PyTorch**。

## 来源与版权

- **模型权重**：DeepFilterNet3 低延迟版（DF3-LL），来自官方仓库
  [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
  （`models/DeepFilterNet3_ll_onnx.tar.gz`，MIT/Apache-2.0）。
- **流式化脚本**（`split_encoder_and_patch_decoders.py`、`merge_split_models.py`、
  `patch_onnx_streaming.py`、`merge_onnx.py`、`compare_wav.py`）：取自实时推理项目
  **deepfilter-rt**，按其许可证复制到本目录，仅做最小适配。
- **DSP 公式**（STFT/ERB/归一化/深度滤波/合成）：端口自 DeepFilterNet 的 Rust `libDF`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `build_streaming_onnx.py` | **一键构建**：缺模型时自动下载离线 ONNX → 拆分/打补丁 → 合并 → 安装到 C# runtimes。 |
| `split_encoder_and_patch_decoders.py` | 拆分 `enc.onnx` 为 conv/gru 两段，并给两个解码器打补丁，暴露 GRU 状态 I/O。 |
| `merge_split_models.py` | 把 4 个流式子图合并为单个 `combined_streaming.onnx`（可选 onnxsim 简化）。 |
| `patch_onnx_streaming.py` | 备选方案：直接在 `combined.onnx` 上打补丁（质量略低，保留作参考）。 |
| `merge_onnx.py` | 备选：把 enc/erb_dec/df_dec 合并为一个 `combined.onnx`。 |
| `dump_reference.py` | **对拍参考**：numpy 端口 libDF 的 DSP + 逐帧编排，驱动流式 ONNX，dump 中间张量与增强音频。 |
| `compare_wav.py` | 音频指标对比（相关系数 / SNR / 对齐），用于与 Python / Rust 参考交叉验证。 |
| `requirements.txt` | 依赖清单。 |

## 用法

```bash
# 0) 安装依赖（conda 环境示例）
pip install -r requirements.txt

# 1) 构建有状态流式模型（缺模型时自动从官方仓库下载离线 ONNX）
python build_streaming_onnx.py
#   产物：combined_streaming.onnx + config.ini
#   安装到：../../src/VoiceBox.DNR.DeepFilterNet/runtimes/models/dnr/deepfilternet/dfn3_ll/
#   重建：python build_streaming_onnx.py --force

# 2) 生成对拍向量（若模型缺失会自动触发上一步）
python dump_reference.py                       # 确定性合成信号
python dump_reference.py path/to/48k_mono.wav  # 指定 wav（自动转 48k 单声道）
#   产物：outputs/vectors/{input,enhanced,feat_erb,feat_spec,mask,coefs,lsnr}.bin + meta.json

```

下载地址（可用 `--url` 覆盖为镜像）：
`https://github.com/Rikorose/DeepFilterNet/raw/refs/heads/main/models/DeepFilterNet3_ll_onnx.tar.gz`

## 模型参数（DF3-LL，来自 `config.ini`）

`sr=48000, fft=960, hop=480, freq_bins=481, nb_erb=32, nb_df=96, df_order=5,`
`df_lookahead=0, conv_lookahead=0`（算法延迟 = fft−hop = **480 样本 ≈ 10 ms**）。

## 对拍结论（供回归参考）

C# 实现与本目录 Python 参考逐级对拍：特征/掩码/系数误差 ~1e-6，端到端
`maxErr≈1e-7, corr=1.000000`（本质逐位一致）；在真实带噪语音上 **SI-SDR 提升 ~12.8 dB**。
