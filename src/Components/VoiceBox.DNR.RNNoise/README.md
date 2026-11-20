# VoiceBox.DNR.RNNoise

`VoiceBox.DNR.RNNoise` 是基于 Xiph.Org RNNoise 的**通用实时降噪**后端，用于抑制麦克风采集中的背景噪声。它通过 RNNoise.NET 调用原生库，将 RNNoise 的流式处理接入 VoiceBox 的 `IDigitalNoiseReductionService`。

## 原项目与开源许可证

- 原生降噪库 **RNNoise** 采用 **BSD-3-Clause**。[https://github.com/xiph/rnnoise](https://github.com/xiph/rnnoise) · [上游 COPYING](https://github.com/xiph/rnnoise/blob/main/COPYING)
- .NET 封装 **RNNoise.NET** 采用 **MIT License**，版权声明为 `Copyright (c) 2023 Yellow Dog Man Studios`。[https://github.com/Yellow-Dog-Man/RNNoise.Net](https://github.com/Yellow-Dog-Man/RNNoise.Net) · [上游 LICENSE](https://github.com/Yellow-Dog-Man/RNNoise.Net/blob/main/LICENSE)

## 接入 VoiceBox

组件面向 .NET 10，输入格式为单声道 Float32，默认采样率为 **16 kHz**。RNNoise 在 **48 kHz** 下以 480 个样本（10 ms）为一帧处理，VoiceBox 会在输入采样率与 48 kHz 之间自动转换，并以输入采样率输出。仓库提供 Windows x64、Linux x64 和 Linux ARM64 的原生库。

```csharp
using VoiceBox.Core;
using VoiceBox.DNR.RNNoise;

using var manager = new AudioServiceManager();
manager.AddRNNoiseDNRService(options => options.SampleRate = 16000);
```

使用 MiniAudio 双工采集时，在 `AddMiniAudioDuplexService` 的配置中设置 `EnableDnr = true`。独立处理时，可通过 `manager.GetDNRServiceInstance()` 获取服务。

| `RNNoiseOptions` | 用途 |
| --- | --- |
| `SampleRate` | 实际输入采样率，默认 `16000`。 |
| `ModelPath` | 自定义 `.rnnn` 模型路径。留空时使用原生库的内置模型。 |
| `Bypass` | 设为 `true` 时原样透传，可用于试听对比。默认 `false`。 |

`Process(Span<float>)` 消费整块输入，把有效输出写入缓冲区开头，并返回有效样本数。启动时返回值可能为 `0`。每个实例维护一路音频的状态，换流时调用 `Reset()`。完整录音可用 `VoiceBox.Core.Services.DNRService` 中的 `ProcessAll` 扩展方法，取出尾部缓存并返回等长结果。
