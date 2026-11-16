# VoiceBox.TTS.Hojo

Hojo-TTS-Light（80M）和 Hojo-TTS-Light-40M-V2 的原生 C# / ONNX Runtime CPU 实现，同时支持原始权重和各自的 LM INT8 量化版。提供 `ITTSService`、`IStreamTTSService` 和 `IAsyncDisposable`，输出 24kHz 单声道 float32 PCM，不依赖 Python 进程进行推理。根据部署包的 `hojo.json` 自动选择架构；应用只需切换 `ModelPath`，可通过命名服务注册多个模型实例。

| 源模型 | LM 层数 / 词表 | 固定音色输入 | 本次生成的部署目录 |
|---|---|---|---|
| Hojo-TTS-Light | 17 / 17667 | 离线参考 token + 2048 维向量 | `E:\Temp\tts\Hojo-TTS-Light-dotnet` |
| Hojo-TTS-Light-int8 | 17 / 17667 | 同上 | `E:\Temp\tts\Hojo-TTS-Light-int8-dotnet` |
| Hojo-TTS-Light-40M-V2 | 10 / 17685 | 官方 speaker slots + 192 维向量 | `E:\Temp\tts\Hojo-TTS-Light-40M-V2-dotnet` |
| Hojo-TTS-Light-40M-V2-int8 | 10 / 17685 | 同上 | `E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8-dotnet` |

旧目录 `Hojo-TTS-Light-40M-V2-dotnet-int8` 的部署格式仍兼容。

## 模型准备

部署前执行一次转换，将 CPU 不支持的 BF16 展开为 FP32、拆出数值张量并生成 SHA-256 清单。原版部署包保持未量化权重；INT8 包保留 LM 量化图，FineLocal 和声码器使用 FP32。80M 的合并解码图在原有硬量化边界拆开，无需训练模型。

在仓库根目录执行：

```powershell
python -m pip install -r scripts/hojo-tts-prepare/requirements.txt
python scripts/hojo-tts-prepare/prepare.py --source E:\Temp\tts\Hojo-TTS-Light --output E:\Temp\tts\Hojo-TTS-Light-dotnet --voices-json scripts/hojo-tts-prepare/voices.example.json
python scripts/hojo-tts-prepare/prepare.py --source E:\Temp\tts\Hojo-TTS-Light-int8 --output E:\Temp\tts\Hojo-TTS-Light-int8-dotnet --voices-json scripts/hojo-tts-prepare/voices.example.json
python scripts/hojo-tts-prepare/prepare.py --source E:\Temp\tts\Hojo-TTS-Light-40M-V2 --output E:\Temp\tts\Hojo-TTS-Light-40M-V2-dotnet
python scripts/hojo-tts-prepare/prepare.py --source E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8 --output E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8-dotnet
```

转换目标必须是新目录，已生成的目录可以直接使用；原始模型不会修改。尚未量化时可使用 `scripts/hojo-tts-benchmark/prepare_int8.py` 生成两种 LM INT8 源包。Python 只用于离线准备，生产服务使用生成的完整目录：

```text
hojo.json
lm.onnx                 # FP32 或动态 INT8 LM，包含 KV cache 输入/输出
fine.onnx               # FP32 FineLocal
decoder.onnx            # FP32 codec
embeddings.f32          # little-endian float32
speakers.f32           # 仅 40M V2
speaker_vectors.f32
voice_prompts.json     # 仅 80M
tokenizer.json
LICENSE.txt
```

应用引用 `Microsoft.ML.OnnxRuntime` CPU 原生包，版本与仓库中央版本一致。库引用 Managed 包，避免强制应用引入 GPU 运行库。中文/英文规范化使用仓库的 WeTextProcessing.NET 及随附 FST、原生库；发布时应保留其 `runtimes` 资源目录。四种模型均通过 Windows x64 真实模型验证，40M V2 INT8 另有 Linux x64 四核实测。

80M 原版没有官方固定音色列表，`voice.npz` 只有词嵌入。`voices.example.json` 将仓库 VoxCPM 示例中的小小女声、四川女声、远方男声、四川男声依次固定为 `hojo_zh_f_01`、`hojo_zh_f_02`、`hojo_zh_m_01`、`hojo_zh_m_02`。这些 ID 与 V2 一致以便切换模型，但**不是 V2 的相同说话人**；`ListVoices()` 的 Description 标明参考来源。生产部署只包含离线计算的参考 codes/文本和说话人向量，不需要 WAV、codec encoder 或 speaker encoder，也不提供运行时克隆接口。可修改 JSON 后重新准备自己的固定音色包。

## 使用

```csharp
using Microsoft.Extensions.Options;
using VoiceBox.TTS.Hojo;

await using var tts = new HojoTTSService(Options.Create(new HojoTTSOptions
{
    ModelPath = @"E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8-dotnet",
    DefaultVoice = "hojo_zh_f_01",
    ThreadsNum = 2,
    ChunkFrames = 40,
}));

await foreach (var chunk in tts.GetStreamAudioAsync("今天下午可能会下雨，出门记得带伞。", ct: cancellationToken))
{
    // PCM 数组归消费方所有，后续推理不会修改它。
    // 播放器应使用 24000Hz / mono / float32，并实施自己的有界缓冲和背压。
    if (chunk.Samples.Length > 0)
        await PlayPcmAsync(chunk.Samples, cancellationToken);
}

var whole = await tts.GetAudioAsync("你好。", ct: cancellationToken);
```

`PlayPcmAsync` 表示应用自己的播放操作。文本输入和 PortAudio 流式播放示例见 [HojoTtsDemo](../Demos/HojoTtsDemo/README.md)。

DI 注册与 VoxCPM 保持一致：

```csharp
services.AddHojoTTSService(options => options.ModelPath = modelDirectory);

// AudioServiceManager 支持默认和命名实例，ITTSService 与 IStreamTTSService 指向同一个实例。
manager.AddHojoTTSService(options => options.ModelPath = modelDirectory, "hojo");
```

第一次合成或 `ListVoices()` 时加载模型并检查包完整性；加载失败可重试。为了避免首次对话承担初始化开销，启动时先调用 `ListVoices()`，再预热一句所需语言的文本。中文默认音色 `hojo_zh_f_01`；可通过 `ListVoices()` 查询，四种中文音色为 `hojo_zh_f_01`、`hojo_zh_f_02`、`hojo_zh_m_01`、`hojo_zh_m_02`。

## 流式与性能

完整文本输入 → 自回归 LM（原生 KV 复用）→ FineLocal（60 帧历史）→ codec（8 帧前瞻）→ ISTFT（2 帧边界上下文）→ PCM。

- 默认 `ChunkFrames=40`：常规块为 800ms 音频，收尾块包含剩余前瞻，可更短或略长；`20` 对应 400ms，首块更早但重复计算更多。
- 使用固定大小的 FineLocal / codec 窗口；不会累计整句 hidden、bits 或波形。LM KV 随上下文增长，但受 4096 位置和 `MaxNewFrames` 限制。
- 首块还需十帧前瞻。输入短于一个块时，在 EOS 后直接输出尾部。
- 正常结束后发送一个空 `IsFinal=true` 块；取消、推理失败、达到生成上限时抛异常，不伪装成成功。
- 这是音频输出流式；LLM 的增量文字需要先缓冲成短句再提交。单句内部不支持持续追加文本。
- 默认关闭空闲 ORT 线程自旋，避免三个顺序执行的会话争抢 CPU。设置自定义 `SessionOptionsFactory` 时由调用方负责这些选项。
- CPU 测量与验证说明见 [VALIDATION.md](VALIDATION.md)，完整 benchmark 可从 [测试项目](../Test/VoiceBox.TTS.Hojo.Benchmark/README.md) 复现。RTF、首块就绪时间不等于声卡端延迟。

## 并发、取消与资源生命周期

每个服务实例同时运行一次推理。默认最多接纳 8 个活动/等待请求，超出容量立即抛 `InvalidOperationException`。播放队列默认只有 2 块，慢消费者使生产者等待，不丢音频、不无限积压。实时对话应取消已经过时的回答，而非不断排入旧文本。

请求参数在调用时复制。每个请求独立持有 RNG、RunOptions、KV 和分块状态；取消通过 `RunOptions.Terminate` 中断正在执行的 ORT。`await foreach` 提前 `break` 会取消并等待生产者退出，之后才释放请求资源。务必正常释放异步枚举器。

`DisposeAsync()` 拒绝新请求，取消活动/等待请求，并等待模型加载与原生调用退出后释放会话。不会在超时后强行释放仍被原生推理访问的内存。同步 `Dispose()` 兼容公共接口；异步宿主优先使用 `await using`。

`OutputSampleRate`、`OutputChannels` 是固定格式，不触发模型加载。服务不提供克隆、情绪指令或速度变换；设置不支持的参数会明确报错。

## 配置边界

| 选项 | 默认 | 说明 |
|---|---:|---|
| ThreadsNum | 2 | ORT intra-op 线程，1–128 |
| ChunkFrames | 40 | 5–100 帧，每帧 20ms |
| MaxNewFrames | 1500 | 生成 token 护栏；调用参数只能收紧，不能放宽 |
| MaxTextLength | 2048 | 输入 UTF-16 字符数上限；还需满足模型上下文限制 |
| MaxPendingRequests | 8 | 活动和等待请求的总容量 |
| BufferedChunks | 2 | PCM 通道容量，1–16 |
| NormalizeText | true | WeText 中英文规范化，保护 `<zh_...>` 拼音标签 |
| DefaultSeed | 42 | 同版本、同参数可复现；null 每次随机 |
| Temperature / TopP / RepetitionPenalty | 0.8 / 0.95 / 1.1 | 可通过 TtsSynthesisOptions 的 Audio 参数覆盖 |

支持 `Voice`、`Seed`、`DoSample`、`MaxNewFrames`、`AudioTemperature`、`AudioTopK`、`AudioTopP`、`AudioRepetitionPenalty`、`Language`。`DoSample=false` 或温度为 0 使用贪心。C# 与 NumPy 随机数实现不同，同一 seed 不保证跨语言采样结果一致；分词及固定 LM token/hidden 到音频的路径已单独对照。

80M 固定参考占用额外 LM 上下文。部分参考音色在贪心模式下会重复，建议保留默认采样；生成上限会明确报错，应用应取消过时回答并按短句提交。

WeTextProcessing.NET 使用其随附 FST，版本可能不同于 Python 调研环境；日期、专有名词与多音字仍需应用侧审核。固定提示语可缓存 PCM。

## 测试

```powershell
# 无大模型的 CI：真实模型用例会明确显示为 skipped。
dotnet test src/Components/Test/VoiceBox.TTS.Hojo.Tests -c Release

# 在测试机器上生成独立 Python 对照数据；使用 prepare/requirements.txt 的 ORT 1.31.1。
python scripts/hojo-tts-prepare/create_golden.py --model-dir E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8 --output E:\Temp\tts\hojo-golden
$env:HOJO_MODEL_PATH='E:\Temp\tts\Hojo-TTS-Light-40M-V2-int8-dotnet'
$env:HOJO_GOLDEN_PATH='E:\Temp\tts\hojo-golden'
dotnet test src/Components/Test/VoiceBox.TTS.Hojo.Tests -c Release
```

INT8 LM 的严格 token/hidden 对照需要 Python 与 C# 使用相同模型、ORT 版本和 CPU 平台。不同量化内核可能改变贪心 token；更新运行库或更换机器后重新生成 golden，版本与模型哈希记录在 `provenance.json`。

80M 的 golden 命令需加 `--voices-json scripts/hojo-tts-prepare/voices.example.json`，与部署时的参考定义保持一致。测试使用官方生成 token 逐步驱动 C# LM，对照完整 logits 和对应 hidden，避免将两种 RNG 的采样差异当成推理错误。四个源模型分别生成 golden，并将 `HOJO_MODEL_PATH`、`HOJO_GOLDEN_PATH` 指向对应的一对目录运行同一套测试。

模型权重遵循 Hojo 官方 Apache-2.0 许可；离线准备脚本引用的官方辅助代码保留原始许可证。C# 运行时为本仓库独立实现。
