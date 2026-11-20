# VoiceBox.DNR.DeepFilterNet

`VoiceBox.DNR.DeepFilterNet` 是基于 **DeepFilterNet3 低延迟版（DF3-LL）** 的通用流式降噪和语音增强后端。模型利用 ERB 频带增益和复数谱深度滤波减轻麦克风录音中的背景噪声。

## 原项目与开源许可证

- 原项目 **DeepFilterNet** 的 GitHub 地址为 [https://github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)。
- 上游代码和模型权重双重授权，可选择 **MIT License** 或 **Apache License 2.0**。[上游许可证说明](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE) · [MIT 原文](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE-MIT) · [Apache-2.0 原文](https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE-APACHE)

## 模型与音频格式

VoiceBox 从上游 `DeepFilterNet3_ll_onnx.tar.gz` 构建 `combined_streaming.onnx`，将 GRU 状态暴露为模型输入和输出。C# 服务逐帧推理并维护跨帧状态。模型和配置位于 `runtimes/models/dnr/deepfilternet/dfn3_ll/`。[VoiceBox 模型导出说明](../../../scripts/DeepFilterNet-onnx-export/README.md)

输入为单声道 Float32，`SampleRate` 默认 **48 kHz**。内置模型使用 960 点分析窗、480 样本步长，每 10 ms 推理一次。输入采样率与模型不同时，组件在内部重采样，并以输入采样率输出。`FrameSize` 与 `DelaySamples` 均以输入采样率下的样本数表示。

## 接入 VoiceBox

组件面向 .NET 10。CPU 为默认执行提供者，`ThreadsNum` 默认为 `2`。宿主应用使用 CPU 时需引用 `Microsoft.ML.OnnxRuntime`，使用 CUDA 时需提供启用 CUDA 的 ONNX Runtime 和匹配的运行库。

```csharp
using VoiceBox.Core;
using VoiceBox.DNR.DeepFilterNet;

using var manager = new AudioServiceManager();
manager.AddDeepFilterNetDNRService(options => options.SampleRate = 48000);
```

`Model` 默认为 `DeepFilterNetModel.Dfn3LowLatency`。如需指定自定义 `ModelDir`，该目录应包含与服务兼容的 `combined_streaming.onnx` 和 `config.ini`。

MiniAudio 双工采集需设置 `EnableDnr = true`。独立处理时，可用 `manager.GetDNRServiceInstance()` 获取服务，以 `Process(Span<float>)` 返回的样本数读取缓冲区开头的有效输出。换流时调用 `Reset()`。完整录音可使用 `VoiceBox.Core.Services.DNRService` 中的 `ProcessAll`。[麦克风实时降噪 Demo](../Demos/DeepFilterNetLiveDemo)
