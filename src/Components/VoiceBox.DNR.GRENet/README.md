# VoiceBox.DNR.GRENet

**抑制房间混响（dereverberation）** 是 `VoiceBox.DNR.GRENet` 在 VoiceBox 中的主要用途。它在麦克风采集链路中实时处理语音，减轻墙壁、天花板等表面反射造成的拖尾，让远场拾音中的人声更清晰。

组件通过 `IDigitalNoiseReductionService` 注册为 VoiceBox 的 DNR 后端，处理单路采集音频。VoiceAssistantDemo 的双工配置先由 WebRTC APM 抑制背景噪声，并利用扬声器播放参考执行声学回声消除（AEC），再由 GRE-Net 处理录音中的房间混响。[VoiceAssistantDemo 的 Plan04 配置](https://github.com/withsalt/voicebox/blob/main/src/Assistant/Demos/VoiceAssistantDemo/Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.cs)

## 原项目与开源许可证

- 原项目 **GRE-Net-Streaming** 的 GitHub 地址为 [https://github.com/wangchengzhong/GRE-Net-Streaming](https://github.com/wangchengzhong/GRE-Net-Streaming)。
- 原项目采用 **MIT License**，版权声明为 `Copyright (c) 2026 wangchengzhong`。VoiceBox 的 NuGet 包附带完整的 `LICENSE`。[上游许可证原文](https://github.com/wangchengzhong/GRE-Net-Streaming/blob/master/LICENSE)

## 模型来源

上游研究联合建模语音的幅度和相位，并在相位分支采用全局旋转等变性。GRE-Net-Streaming 提供逐帧推理所需的因果模型和流式缓存。[GRE-Net 研究仓库](https://github.com/wangchengzhong/GRE-Net) · [研究论文](https://arxiv.org/abs/2602.08556)

VoiceBox 使用上游 `cp_model_casual_whamr/g_00515000` checkpoint 重新导出静态 FP32 `stream.onnx`，由 C# 服务维护跨帧状态。同一份模型用于 CPU 和 CUDA，随组件打包到 `runtimes/models/dnr/grenet/stream.onnx`。

## 接入 VoiceBox

组件面向 .NET 10，宿主应用需提供带原生库的 ONNX Runtime 包。

| 执行提供者 | 宿主应用的 NuGet 依赖 | 默认 `ThreadsNum` |
| --- | --- | --- |
| CPU（默认） | `Microsoft.ML.OnnxRuntime` | 2 |
| CUDA（Windows、Linux） | `Microsoft.ML.OnnxRuntime.Gpu`，另需兼容的 NVIDIA 驱动、CUDA 和 cuDNN | 1 |

```csharp
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.DNR.GRENet;

using var manager = new AudioServiceManager();
manager.AddGRENetDNRService(options => options.ExecutionProvider = OnnxExecutionProvider.Cpu);
```

通过音频后端（如果MiniAudio）双工采集时，在 `AddMiniAudioDuplexService` 的配置中设置 `EnableDnr = true`。直接处理音频时，可以通过 `manager.GetDNRServiceInstance()` 获取服务。也可以在 `IServiceCollection` 上调用 `AddGRENetDNRService` 来注册默认 DNR 服务。

`GRENetOptions` 的音频格式固定为 **16 kHz、单声道、Float32**。`Model` 留空时加载随包提供的模型，指定路径时需使用与组件流式接口兼容的 ONNX 模型。

## 流式处理

- 模型使用 400 个样本（25 ms）的分析窗，每 200 个样本（12.5 ms）推理一次。`FrameSize` 和 `DelaySamples` 均为 200，分别表示建议输入块大小和标称启动缓冲量。
- 直接调用 `Process(Span<float>)` 时，可以传入任意长度的音频块。方法消费整块输入，返回写入缓冲区开头的有效输出样本数。启动阶段返回值可能为 `0`。
- 每个实例维护一路音频的流式状态，开始新流时调用 `Reset()`。处理完整录音时，可使用 `VoiceBox.Core.Services.DNRService` 中的 `ProcessAll` 扩展方法，自动取出尾部缓存并返回与输入等长的结果。

## 重新导出模型

将上游仓库检出到 `scripts/grenet-onnx-export/GRE-Net-Streaming` 后，在 VoiceBox 仓库根目录执行以下命令。验证脚本将导出模型的逐帧输出与上游 PyTorch checkpoint 对照。

```sh
python -m pip install torch numpy onnx onnxsim onnxruntime joblib pesq
python scripts/grenet-onnx-export/export.py --source scripts/grenet-onnx-export/GRE-Net-Streaming --output ./stream.onnx
python scripts/grenet-onnx-export/validate.py --source scripts/grenet-onnx-export/GRE-Net-Streaming --model ./stream.onnx
```
