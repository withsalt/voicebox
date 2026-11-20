# VoiceBox.DNR.DPDFNet

`VoiceBox.DNR.DPDFNet` 将 Ceva 的 **DPDFNet** 接入 VoiceBox，用于通用实时降噪和语音增强。VoiceBox 集成了四档 16 kHz 流式模型，可按部署设备的算力和降噪需求选择。

## 原项目与开源许可证

- 原项目 **DPDFNet** 的 GitHub 地址为 [https://github.com/ceva-ip/DPDFNet](https://github.com/ceva-ip/DPDFNet)。上游在 DeepFilterNet2 的基础上引入双路径 RNN，并提供预训练模型及 ONNX 导出。[官方模型仓库](https://huggingface.co/Ceva-IP/DPDFNet)
- 上游项目和官方模型仓库采用 **Apache License 2.0**。[上游 LICENSE](https://github.com/ceva-ip/DPDFNet/blob/main/LICENSE)

## 内置模型

| `DPDFNetModel` | ONNX 文件 |
| --- | --- |
| `Baseline` | `baseline.onnx` |
| `DpDfNet2`（默认） | `dpdfnet2.onnx` |
| `DpDfNet4` | `dpdfnet4.onnx` |
| `DpDfNet8` | `dpdfnet8.onnx` |

内置模型位于 `runtimes/models/dnr/dpdfnet/`。输入固定为 **16 kHz、单声道、Float32**。内置模型使用 320 点分析窗、160 样本步长（10 ms）。`FrameSize` 和 `DelaySamples` 均为 160。若通过 `ModelPath` 自定义模型，需使用符合组件流式接口的 16 kHz ONNX 导出。

## 接入 VoiceBox

组件面向 .NET 10。CPU 为默认执行提供者，`ThreadsNum` 默认为 `2`。宿主应用使用 CPU 时需引用 `Microsoft.ML.OnnxRuntime`，使用 CUDA 时需提供启用 CUDA 的 ONNX Runtime 和匹配的运行库。

```csharp
using VoiceBox.Core;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.DNR.DPDFNet.Models;

using var manager = new AudioServiceManager();
manager.AddDPDFNetDNRService(options => options.Model = DPDFNetModel.DpDfNet2);
```

`AttenuationLimitDb` 可限制降噪衰减。默认的 `MagnitudeFloor` 模式会保留输入相位，对模型给出的幅度增益施加下限。留空时幅度增益下限为零。

MiniAudio 双工采集需设置 `EnableDnr = true`。独立处理时，可用 `manager.GetDNRServiceInstance()` 获取服务，以 `Process(Span<float>)` 返回的样本数读取缓冲区开头的有效输出。换流时调用 `Reset()`。完整录音可使用 `VoiceBox.Core.Services.DNRService` 中的 `ProcessAll`。[麦克风实时降噪 Demo](../Demos/DPDFNetLiveDemo)
