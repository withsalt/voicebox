# DTLN-aec ONNX 导出

本目录用于把上游 [`DTLN-aec`](DTLN-aec/README.md) 预训练 TFLite 模型，稳定复现地导出为 ONNX。

导出的 ONNX 模型是流式模型。运行时由宿主持有循环状态，每次输入 8 ms hop：

- 采样率：16 kHz mono
- 分析窗：512 samples
- hop size：128 samples
- 模型分两段：第一段预测频域幅度 mask，第二段做时域增强

## 环境

使用已经准备好的 conda 环境，或创建兼容的 Python 3.12 环境：

```powershell
conda activate dtln-aec-export
python -m pip install -r requirements.txt
```

上游 `DTLN-aec/requirements.txt` 为原始 TFLite demo 固定了 TensorFlow 2.3.1。本导出项目使用较新的 TensorFlow，只用于驱动 `tf2onnx` 和在 Python 3.12 上做 TFLite 对齐验证。

## 导出

在当前目录执行：

```powershell
conda activate dtln-aec-export
python export_dtln_aec_onnx.py --force
```

默认输入：

- `DTLN-aec/pretrained_models/dtln_aec_{128,256,512}_{1,2}.tflite`

如果上游 `DTLN-aec` 目录不存在，导出脚本会先从 `https://github.com/breizhn/DTLN-aec.git` 自动 clone。若目录已经存在但请求的预训练文件缺失，脚本会提前失败，避免悄悄覆盖本地不完整 checkout。

自动 clone 是非交互式的，并带有超时；如果代理或 GitHub 连通性有问题，会明确失败，而不是无限挂起。

默认输出：

- `onnx_out_named/dtln_aec_{128,256,512}_{1,2}.onnx`

导出流程先调用 `tf2onnx.convert --tflite`，再后处理 ONNX 图，把输入输出改成稳定的语义化名称。这样可以避开 `tf2onnx` 在 TFLite 分支中直接使用 `--rename-inputs` 时可能触发的重命名问题。

常用命令：

```powershell
python export_dtln_aec_onnx.py --units 128 --force
python export_dtln_aec_onnx.py --units 128,256 --output-dir onnx_out_named --force
python export_dtln_aec_onnx.py --keep-raw --force
python export_dtln_aec_onnx.py --no-clone --force
python export_dtln_aec_onnx.py --upstream-url https://github.com/breizhn/DTLN-aec.git --force
python export_dtln_aec_onnx.py --clone-timeout-sec 600 --force
```

## 验证

运行 ONNX checker、ONNX Runtime 加载、零输入推理，以及 TFLite/ONNX 流式一致性验证：

```powershell
conda activate dtln-aec-export
python validate_dtln_aec_onnx.py --frames 100
```

只做快速 ONNX 检查：

```powershell
python validate_dtln_aec_onnx.py --skip-parity
```

一致性验证默认使用上游 double-talk 样例：

```text
DTLN-aec/audio_samples/DMTgmZwtgUilp4omPK7-OQ_doubletalk_mic.wav
DTLN-aec/audio_samples/DMTgmZwtgUilp4omPK7-OQ_doubletalk_lpb.wav
```

## 流式接口

第一段：

```text
mic_mag    [1,1,257]
state1_in  [1,2,units,2]
lpb_mag    [1,1,257]
  -> mask       [1,1,257]
  -> state1_out [1,2,units,2]
```

第二段：

```text
estimated_block [1,1,512]
state2_in       [1,2,units,2]
lpb_block       [1,1,512]
  -> enhanced_block [1,1,512]
  -> state2_out     [1,2,units,2]
```

`units` 可取 `128`、`256` 或 `512`。

## 宿主运行时注意事项

ONNX 模型本身不包含 FFT、iFFT、滚动输入缓冲和 overlap-add。宿主运行时需要维护：

- 512-sample 麦克风滚动缓冲
- 512-sample loopback/远端参考滚动缓冲
- `state1` 和 `state2`
- real FFT / inverse real FFT
- 频域 `mask` 应用
- overlap-add 输出，每个 hop 产出 128 samples

接入 VoiceBox 时，可以参考现有 `VoiceBox.AEC.DFSMN` 服务里的实时 FIFO、ONNX Runtime session 持有方式和 STFT 类处理结构。
