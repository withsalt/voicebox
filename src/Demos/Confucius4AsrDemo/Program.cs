using Microsoft.Extensions.Logging;
using VoiceBox.ASR.Confucius4;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.ASRService.Models;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.RNNoise;
using VoiceBox.VAD.FireRed;

namespace Confucius4AsrDemo;

internal static class Program
{
    private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
    private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "asr", "Confucius4-R2T2", "fp16");
    private static readonly OnnxExecutionProvider OnnxProvider = OnnxExecutionProvider.Cpu;

    public static async Task<int> Main()
    {
        try
        {
            await RunAsync();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static async Task RunAsync()
    {
        using var audioServiceProxy = new AudioServiceManager()
            .AddLogger(b =>
            {
                b.SetMinimumLevel(LogLevel.Information);
                b.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss ";
                });
            })
            .AddPortAudioRecordService(options =>
            {
                options.SampleRate = 16000;
                options.ChannelCount = 1;
                options.SampleFormat = SampleFormat.Float32;
            })
            .AddFireRedVoiceActivityDetectService(options =>
            {
                options.SampleRate = 16000;
                options.ChannelCount = 1;
                options.SampleFormat = SampleFormat.Float32;
                options.ExecutionProvider = OnnxProvider;
            })
            .AddRNNoiseDNRService(options =>
            {
                options.SampleRate = 16000;
                options.ChannelCount = 1;
                options.SampleFormat = SampleFormat.Float32;
            })
            .AddConfucius4ASRService(options =>
            {
                options.ModelPath = ModelPath;
                options.ExecutionProvider = OnnxProvider;
                options.Language = "Chinese";
                options.ChunkMilliseconds = 160;
            })
            .Build();

        // 服务由 AudioServiceManager 统一释放，先停止录音并等待全部会话收尾。
        var recordService = audioServiceProxy.GetAudioRecordInstance();
        var vadService = audioServiceProxy.GetVoiceActivityDetectInstance();
        var dnrService = audioServiceProxy.GetDNRServiceInstance();
        var asrService = audioServiceProxy.GetStreamASRServiceInstance();

        var inputDevices = recordService.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(device => device.Id).ToList();
        Console.WriteLine("可用输入设备:");
        foreach (var device in inputDevices ?? [])
            Console.WriteLine($"{device.Id}. {device.Name}{(device.IsDefault ? "【默认设备】" : "")}");
        Console.WriteLine();

        var sessionLock = new object();
        RecognitionSession? currentSession = null;
        var finishingSessions = new HashSet<Task>();
        var stopRequested = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Exception? pipelineFailure = null;

        void ReportFailure(string message, Exception exception)
        {
            Console.Error.WriteLine($"{message}: {exception.Message}");
            Interlocked.CompareExchange(ref pipelineFailure, exception, null);
            stopRequested.TrySetResult();
        }

        async Task ReadResultsAsync(string sessionId, IAsyncEnumerable<AsrResult> results)
        {
            try
            {
                await foreach (var result in results)
                {
                    if (!string.IsNullOrWhiteSpace(result.Text))
                        Console.WriteLine($"[实时][{sessionId}] {result.Text}");
                }
            }
            catch (Exception ex)
            {
                ReportFailure($"[ASR错误][{sessionId}]", ex);
            }
        }

        async Task FinishSessionAsync(RecognitionSession session)
        {
            try
            {
                var result = await asrService.FinishSessionAsync(session.Id);
                await session.Results;
                Console.WriteLine($"\n=== 本轮识别结果 [{session.Id}] ===");
                Console.WriteLine($"文本: {result?.Text ?? "(空)"}");
                Console.WriteLine("====================\n");
            }
            catch (Exception ex)
            {
                ReportFailure($"结束ASR会话失败 [{session.Id}]", ex);
                await session.Results;
            }
        }

        void FinishCurrentSession()
        {
            lock (sessionLock)
            {
                if (currentSession is not { } session)
                    return;

                currentSession = null;
                finishingSessions.RemoveWhere(task => task.IsCompleted);
                finishingSessions.Add(FinishSessionAsync(session));
            }
        }

        vadService.OnSpeechStarted += (_, _) =>
        {
            try
            {
                lock (sessionLock)
                {
                    if (stopRequested.Task.IsCompleted || currentSession is not null)
                        return;

                    Console.WriteLine("检测到说话开始...");
                    // Confucius4 创建会话同步完成；在返回 VAD 回调前接好首帧和结果流。
                    var session = asrService.StartSessionAsync().GetAwaiter().GetResult();
                    // 必须在 Finish 前取得枚举，不能延迟到后台任务里才订阅。
                    var results = asrService.GetStreamResultAsync(session.SessionId);
                    currentSession = new RecognitionSession(session.SessionId, ReadResultsAsync(session.SessionId, results));
                    Console.WriteLine($"ASR会话已启动: {session.SessionId}");
                }
            }
            catch (Exception ex)
            {
                ReportFailure("启动ASR会话失败", ex);
            }
        };

        vadService.OnSpeaking += (_, e) =>
        {
            try
            {
                lock (sessionLock)
                {
                    if (currentSession is { } session)
                        asrService.FeedAudio(session.Id, e.Data);
                }
            }
            catch (Exception ex)
            {
                ReportFailure("发送ASR音频失败", ex);
            }
        };

        vadService.OnSpeechEnded += (_, e) =>
        {
            Console.WriteLine($"检测到说话结束，时长 {e.Duration?.TotalSeconds:F1}s");
            // 结束事件中的 Data 是整句音频；OnSpeaking 已送入所有分块，无需重复发送。
            FinishCurrentSession();
        };

        recordService.OnDataAvailable += samples =>
        {
            if (stopRequested.Task.IsCompleted)
                return;

            try
            {
                Span<float> denoiseBuffer = samples.Length <= 4096
                    ? stackalloc float[samples.Length]
                    : new float[samples.Length];
                samples.CopyTo(denoiseBuffer);

                int produced = dnrService.Process(denoiseBuffer);
                if (produced > 0)
                    vadService.Predict(denoiseBuffer[..produced]);
            }
            catch (Exception ex)
            {
                ReportFailure("处理录音失败", ex);
            }
        };
        recordService.OnError += (_, e) => ReportFailure("录音失败", e.Exception);
        recordService.OnXrun += (_, e) => Console.Error.WriteLine($"音频缓冲区溢出: {e}");

        ConsoleCancelEventHandler cancel = (_, e) =>
        {
            e.Cancel = true;
            stopRequested.TrySetResult();
        };
        Console.CancelKeyPress += cancel;
        try
        {
            await recordService.StartRecordingAsync();
            Console.WriteLine("开始录音... 按 Q、Esc 或 Ctrl+C 停止录音。");

            while (!stopRequested.Task.IsCompleted)
            {
                if (!Console.IsInputRedirected && Console.KeyAvailable)
                {
                    var key = Console.ReadKey(true).Key;
                    if (key is ConsoleKey.Escape or ConsoleKey.Q)
                        break;
                }

                await Task.WhenAny(stopRequested.Task, Task.Delay(50));
            }
        }
        finally
        {
            try
            {
                Console.WriteLine("正在停止录音...");
                try
                {
                    await recordService.StopRecordingAsync();
                }
                finally
                {
                    // 停止麦克风不会触发 VAD 的结束事件，需主动完成最后一句。
                    FinishCurrentSession();
                    Task[] pending;
                    lock (sessionLock)
                        pending = finishingSessions.ToArray();
                    await Task.WhenAll(pending);
                }
            }
            finally
            {
                Console.CancelKeyPress -= cancel;
            }
        }

        if (pipelineFailure is not null)
            throw new InvalidOperationException("录音或识别失败。", pipelineFailure);
    }

    private sealed record RecognitionSession(string Id, Task Results);
}
