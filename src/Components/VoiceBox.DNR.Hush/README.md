# VoiceBox.DNR.Hush

`VoiceBox.DNR.Hush` 是基于 **Hush（DeepFilterNet3-SE）** 的通用流式降噪和语音增强后端，用于 16 kHz 麦克风采集。上游 Hush 在 DeepFilterNet3 基础上引入背景说话人抑制的训练目标，模型也处理环境噪声。[上游模型说明](https://github.com/pulp-vision/Hush#readme)

## 原项目与开源许可证

- 原项目 **Hush** 的 GitHub 地址为 [https://github.com/pulp-vision/Hush](https://github.com/pulp-vision/Hush)。[官方预训练模型](https://huggingface.co/weya-ai/hush)
- 上游项目和官方模型仓库采用 **Apache License 2.0**。[上游 LICENSE](https://github.com/pulp-vision/Hush/blob/main/LICENSE)

## 模型与音频格式

VoiceBox 将 Hush 模型导出为逐帧 ONNX，把 GRU 状态作为模型输入和输出，由 C# 服务负责跨帧传递。默认模型为 `HushModel.Hush16k`，`combined_streaming.onnx` 与 `config.ini` 位于 `runtimes/models/dnr/hush/hush16k/`。自定义 `ModelDir` 时需要提供兼容的这两个文件。[VoiceBox 流式导出脚本](../../../scripts/hush-onnx-export/export_streaming.py)

输入固定为 **16 kHz、单声道、Float32**。内置模型使用 320 点分析窗，每 160 个样本推理一次（10 ms）。`FrameSize` 和 `DelaySamples` 均为 160。

## 接入 VoiceBox

组件面向 .NET 10。CPU 为默认执行提供者，`ThreadsNum` 默认为 `2`。宿主应用使用 CPU 时需引用 `Microsoft.ML.OnnxRuntime`，使用 CUDA 时需提供启用 CUDA 的 ONNX Runtime 和匹配的运行库。

```csharp
using VoiceBox.Core;
using VoiceBox.DNR.Hush;
using VoiceBox.DNR.Hush.Models;

using var manager = new AudioServiceManager();
manager.AddHushDNRService(options => options.Model = HushModel.Hush16k);
```

MiniAudio 双工采集需设置 `EnableDnr = true`。独立处理时，可用 `manager.GetDNRServiceInstance()` 获取服务，以 `Process(Span<float>)` 返回的样本数读取缓冲区开头的有效输出。换流时调用 `Reset()`。完整录音可使用 `VoiceBox.Core.Services.DNRService` 中的 `ProcessAll`。[麦克风实时降噪 Demo](../Demos/HushDnrLiveDemo)
