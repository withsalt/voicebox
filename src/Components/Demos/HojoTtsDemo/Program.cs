using System.Diagnostics;
using System.Text;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.TTS.Hojo;

namespace HojoTtsDemo;

internal static class Program
{
    // 改为任意一个 Hojo 80M / 40M V2 的 .NET 部署目录，原版和 INT8 均可。
    private const string BASE_MODEL_DIR = @"E:\ML\Audio\voicebox_models\v1";
    private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "tts", "hojo", "Hojo-TTS-Light-40M-V2-int8-dotnet");
    private static readonly OnnxExecutionProvider Provider = OnnxExecutionProvider.Cpu;
    private const string DdefaultVoice = "hojo_zh_f_01";

    private static async Task<int> Main()
    {
        Console.InputEncoding = Encoding.UTF8;
        Console.OutputEncoding = Encoding.UTF8;
        using var stop = new CancellationTokenSource();
        ConsoleCancelEventHandler cancel = (_, e) => { e.Cancel = true; stop.Cancel(); };
        Console.CancelKeyPress += cancel;
        Exception? playbackFailure = null;

        try
        {
            // 1. 注册 TTS 和播放器，由 AudioServiceManager 创建、复用并统一释放。
            using var audio = new AudioServiceManager()
                .AddHojoTTSService(options =>
                {
                    options.ModelPath = ModelPath;
                    options.DefaultVoice = DdefaultVoice;
                    options.ThreadsNum = 2;
                    options.ChunkFrames = 40;
                    options.ExecutionProvider = Provider;
                })
                .AddPortAudioPlayService(options =>
                {
                    options.SampleRate = 24000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                    options.MaxQueueSize = 10;
                    options.MaxFrameSize = 480;
                })
                .Build();

            // 2. 从容器获取服务接口，业务代码不依赖具体实现。
            var tts = audio.GetStreamTTSServiceInstance();
            var player = audio.GetAudioPlayInstance();
            player.OnError += (_, error) =>
            {
                Interlocked.CompareExchange(ref playbackFailure, error.Exception, null);
                stop.Cancel();
            };

            Console.WriteLine("可用音色：" + string.Join(", ", tts.ListVoices().Select(v => v.Name)));
            Console.WriteLine($"当前音色：{DdefaultVoice}。输入文本回车播放，:q 退出，Ctrl+C 取消并退出。");
            try
            {
                await player.StartPlayingAsync(cancellationToken: stop.Token);
                while (!stop.IsCancellationRequested)
                {
                    Console.Write("> ");
                    string? text = await Console.In.ReadLineAsync(stop.Token);
                    if (text is null || text.Trim() == ":q") break;
                    if (string.IsNullOrWhiteSpace(text)) continue;

                    // 3. 合成音频流，并逐块交给播放器。
                    await SynthesizeAndPlayAsync(tts, player, text, stop.Token);
                }
                if (playbackFailure is not null) throw playbackFailure;
            }
            finally
            {
                player.ClearQueue();
                await player.StopPlayingAsync(CancellationToken.None);
            }
            return 0;
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested && playbackFailure is null)
        {
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"失败：{playbackFailure?.Message ?? ex.Message}");
            return 1;
        }
        finally
        {
            Console.CancelKeyPress -= cancel;
        }
    }

    private static async Task SynthesizeAndPlayAsync(IStreamTTSService tts, IAudioPlayService player,
        string text, CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();
        double firstMs = double.NaN;
        long totalSamples = 0;
        int chunks = 0;
        double longestChunkSeconds = 0;
        await foreach (var chunk in tts.GetStreamAudioAsync(text, ct: ct))
        {
            if (chunk.Samples.Length == 0) continue;
            if (chunks == 0) firstMs = stopwatch.Elapsed.TotalMilliseconds;
            chunks++;
            totalSamples += chunk.Samples.Length;

            // 播放器积压时等待，避免长文本合成过快导致队列丢音频。
            while (player.CurrentQueueSize >= 4) await Task.Delay(10, ct);
            if (!player.TryEnqueue(chunk.Samples, out int dropped) || dropped != 0)
                throw new IOException("播放队列已满。");
            longestChunkSeconds = Math.Max(longestChunkSeconds,
                (double)chunk.Samples.Length / tts.OutputSampleRate / tts.OutputChannels);
        }

        // 墙钟包含流式消费中的队列等待，不包含下方的尾部播放等待。
        stopwatch.Stop();
        double durSec = (double)totalSamples / tts.OutputSampleRate / tts.OutputChannels;
        double wallSec = stopwatch.Elapsed.TotalSeconds;
        double rtf = durSec > 0 ? wallSec / durSec : double.NaN;

        while (player.CurrentQueueSize > 0) await Task.Delay(10, ct);
        // 队列长度不包含正在播放的块，等它播完再接收下一句。
        await Task.Delay(TimeSpan.FromSeconds(longestChunkSeconds + 0.15), ct);
        Console.WriteLine($"首包={firstMs,6:F0}ms  时长={durSec,5:F2}s  墙钟={wallSec,5:F2}s  RTF={rtf:F3}  块={chunks}");
    }
}
