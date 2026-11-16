# HojoTtsDemo

最小文本交互示例：通过 `AudioServiceManager` 依赖注入注册 Hojo TTS 服务和播放器，输入文本，通过 `GetStreamAudioAsync` 逐块合成并播放。

在 `Program.cs` 顶部设置 `ModelPath` 和 `Voice`，然后运行：

```powershell
dotnet run --project src/Components/Demos/HojoTtsDemo -c Release
```

默认使用 `E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8-dotnet`，音色为 `hojo_zh_f_01`。`ModelPath` 可切换到 80M / 40M V2 的原版或 INT8 .NET 部署包，详见 [模型准备](../../VoiceBox.TTS.Hojo/README.md)。

输入文本后按回车播放；`:q` 退出，Ctrl+C 取消并退出。空行跳过。程序启动时显示可用音色。

代码只有两个方法：`Main` 使用 `AddHojoTTSService`、`AddPortAudioPlayService` 注册服务并调用 `Build()`，通过 `GetStreamTTSServiceInstance()`、`GetAudioPlayInstance()` 获取服务后读取文本；`SynthesizeAndPlayAsync` 接收 `IStreamTTSService` 和 `IAudioPlayService`，调用 TTS、将 PCM 放入播放队列，并等待尾部播放完毕。

服务由 `AudioServiceManager` 统一释放。队列积压时等待，避免丢音频。播放器按 Hojo 的固定输出格式配置为 24kHz、单声道 Float32，需要可用的默认音频输出设备。

性能测试使用独立的 [Hojo benchmark](../../Test/VoiceBox.TTS.Hojo.Benchmark/README.md)。
