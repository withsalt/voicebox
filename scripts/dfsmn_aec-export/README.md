# DFSMN AEC Streaming ONNX Export

这个目录用于把 `speech_dfsmn_aec_psm_16k` 的 DFSMN AEC 模型导出为**支持流式推理的 ONNX core**。

导出的 ONNX 只负责神经网络核心推理：

- 输入当前帧 fbank 特征和 DFSMN cache
- 输出当前帧频域 mask
- 输出当前帧近端语音概率 `vad_prob`
- 输出更新后的 DFSMN cache

音频侧的 STFT、ISTFT、FIFO、线性 AEC、残余回声抑制由 C# `VoiceBox.AEC.DFSMN` 实现，不放进 ONNX。

## 文件说明

- `export_dfsmn_mask_stream.py`：导出流式 mask+VAD ONNX。
- `verify_dfsmn_mask_stream.py`：验证 ONNX 输出与 PyTorch streaming wrapper 是否一致。
- `modeling_modified/uni_deep_fsmn.py`：ModelScope DFSMN layer 兼容补丁。
- `dependents/speech_dfsmn_aec_psm_16k`：原始 DFSMN AEC 模型文件。
- `dependents/MinDAEC-0.0.2-py3-none-any.whl`：ModelScope AEC pipeline 加载时可能需要的依赖。

## 环境

推荐使用项目里的 `voicebox` conda 环境：

```powershell
conda activate voicebox
```

如果环境缺依赖：

```powershell
pip install -r scripts\dfsmn_aec-export\requirements.txt
pip install scripts\dfsmn_aec-export\dependents\MinDAEC-0.0.2-py3-none-any.whl
```

## 导出

导出到脚本目录下的 `export`：

```powershell
conda run -n voicebox python scripts\dfsmn_aec-export\export_dfsmn_mask_stream.py
```

默认输出：

```text
scripts/dfsmn_aec-export/export/dfsmn_aec_mask_vad_stream.onnx
scripts/dfsmn_aec-export/export/dfsmn_aec_mask_vad_stream.contract.json
scripts/dfsmn_aec-export/export/feature.DEY.mvn.txt
```

直接导出到 C# 运行时目录：

```powershell
conda run -n voicebox python scripts\dfsmn_aec-export\export_dfsmn_mask_stream.py `
  --output src\VoiceBox.AEC.DFSMN\runtimes\models\aec\dfsmn\dfsmn_aec_mask_vad_stream.onnx
```

## 验证

验证默认导出文件：

```powershell
conda run -n voicebox python scripts\dfsmn_aec-export\verify_dfsmn_mask_stream.py --frames 17
```

验证 C# 运行时目录里的 ONNX：

```powershell
conda run -n voicebox python scripts\dfsmn_aec-export\verify_dfsmn_mask_stream.py `
  --onnx src\VoiceBox.AEC.DFSMN\runtimes\models\aec\dfsmn\dfsmn_aec_mask_vad_stream.onnx `
  --frames 17
```

验证通过时会输出类似：

```text
max_abs=0.00000110
mean_abs=0.00000009
vad_max_abs=0.00000030
vad_mean_abs=0.00000013
```

## ONNX 输入输出

默认导出的模型约定：

- `feat`: `float32[1,1,240]`
- `cache_in`: `float32[12,1,512,19]`
- `mask`: `float32[1,1,321]`
- `vad_prob`: `float32[1,1,1]`
- `cache_out`: `float32[12,1,512,19]`

对应音频参数：

- sample rate: `16000`
- frame length: `640`
- hop length: `320`
- feature bins: `80`
- mask bins: `321`

对应 contract metadata 会额外包含：

- `outputs`: `mask,vad_prob,cache_out`
- `vad_dim`: `1`
