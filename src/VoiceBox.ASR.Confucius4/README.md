# VoiceBox.ASR.Confucius4

Confucius4-R2T2 本地 ONNX 流式 ASR，目标框架 .NET 10。项目内包含独立的 ONNX tensor、Tokenizer、音频特征和 DSP 实现，以及 Confucius4 的包校验、Token 回退、稳定结果提交和会话生命周期管理。

```csharp
services.AddConfucius4ASRService(o =>
{
    o.ModelPath = "/models/confucius4-fp16";
    o.ExecutionProvider = OnnxExecutionProvider.Cuda;
    o.Language = "Chinese"; // null：自动检测
});

var asr = provider.GetRequiredService<IStreamASRService>();
var session = await asr.StartSessionAsync();
var results = asr.GetStreamResultAsync(session.SessionId);
var consumer = Task.Run(async () =>
{
    await foreach (var result in results)
        Console.WriteLine(result.CommittedText);
});
asr.FeedAudio(session.SessionId, mono16kFloatSamples);
var final = await asr.FinishSessionAsync(session.SessionId);
await consumer;
```

必须在 Finish 前取得结果枚举。每会话仅允许一个流式消费者；输出是累计快照，慢消费者可能跳过中间快照，但最终快照保留全部文本。`FeedAudio` 复制采样，返回后调用者可复用输入。浮点 PCM 应为单声道、16kHz，通常在 [-1, 1]。使用外部重采样器处理其他格式。

`Confucius4ASRService` 还提供 `TranscribeAsync`（单窗口离线识别）、`CancelSessionAsync` 和 `DisposeAsync`。取消 Finish 的等待不会取消后台识别，可再次调用 Finish；主动终止使用 Cancel。关闭服务先停止接收并等待当前原生调用退出，再释放 ONNX 资源。单个服务用信号量串行访问共享缓存，多会话会产生排队；需要并行吞吐时部署多个实例并根据显存实测容量。

默认 `SuppressOrtWarnings=true` 会隐藏 ORT 已知的 shape/EP placement、Memcpy 和 ScatterND 诊断警告；这些节点仍可能按设计在 CPU 上运行。设置 `SuppressOrtWarnings=false` 并设置 `ORT_LOG_VERBOSE=1` 可查看完整节点分配。

部署宿主引用 `Microsoft.ML.OnnxRuntime.Gpu` 或 CPU native runtime，版本应与 Managed API 匹配。GPU 还需要对应 CUDA/cuDNN 动态库。Python 无需随产品部署。INT8 使用 ORT 1.28 的 8 位权重算子，已验证 CUDA，CPU 能执行但明显较慢；实时场景优先 FP16 CUDA。`VerifyHashesOnLoad=true` 可验证模型包 SHA-256（增加启动读盘开销）。

默认 16 秒窗口限制音频和 KV 缓存资源，长音频达到窗口上限会完成当前段；累计文本随会话增长。建议由 VoiceBox VAD 划分自然语句，以免硬切分降低跨边界单词的准确率。队列满会抛异常，不会静默丢帧。`MaxConcurrentSessions` 是资源保护上限，不是实时并发承诺。

Demo 参考 `Qwen3ASRDemo`，通过 `AudioServiceManager` 组合 PortAudio 麦克风、RNNoise 降噪、FireRed VAD 和 Confucius4 流式识别。输入为 16 kHz 单声道 Float32；VAD 检测到说话后创建会话，实时打印累计识别文本，句末输出最终结果。

先按本机环境修改 `src/Demos/Confucius4AsrDemo/Program.cs` 顶部的 `ModelPath` 和 `OnnxProvider`。默认使用仓库导出的 `scripts/Confucius4-R2T2-onnx-export/outputs/fp16` 模型和 CUDA，模型路径采用当前开发机的绝对路径。运行时使用 NuGet 提供的 ONNX Runtime 1.28.0 GPU native 库，CUDA/cuDNN 依赖需在 PATH 中可用。

```powershell
dotnet run --project src/Demos/Confucius4AsrDemo -c Release
```

启动后列出输入设备并使用默认麦克风。按 Q、Esc 或 Ctrl+C 停止录音，程序等待当前及正在收尾的会话完成后退出。语言和推理分块在 `AddConfucius4ASRService` 中配置，默认中文、160 ms。示例不再使用原来的 WAV/离线/JSON 命令行参数。
