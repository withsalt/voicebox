using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;
using VoiceBox;
using VoiceBox.ASR.Qwen3;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService.Models;
using VoiceBox.DNR.RNNoise;
using VoiceBox.VAD.FireRed;

namespace Qwen3ASRDemo
{
    internal class Program
    {
        private static string? _currentSessionId;
        private static readonly object _sessionLock = new();

        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "asr", "qwen3", "qwen3-asr-0.6B-onnx-fp16");
        private static readonly OnnxExecutionProvider onnxProvider = OnnxExecutionProvider.Cpu;

        static async Task Main(string[] args)
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
                    options.ExecutionProvider = onnxProvider;
                })
                .AddRNNoiseDNRService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                })
                .AddQwen3ASRService(options =>
                {
                    options.ModelPath = ModelPath;
                    options.ExecutionProvider = onnxProvider;
                    options.ChunkSizeSec = 3.0;
                })
                .Build();

            using var recordService = audioServiceProxy.GetAudioRecordInstance();
            using var vadService = audioServiceProxy.GetVoiceActivityDetectInstance();
            using var dnrService = audioServiceProxy.GetDNRServiceInstance();
            using var asrService = audioServiceProxy.GetStreamASRServiceInstance();

            var inputDevices = recordService.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(s => s.Id).ToList();
            Console.WriteLine("可用输入设备:");
            foreach (var item in inputDevices ?? new())
            {
                Console.WriteLine($"{item.Id}. {item.Name}{(item.IsDefault ? "【默认设备】" : "")}");
            }
            Console.WriteLine();

            vadService.OnSpeechStarted += async (object? sender, EventArgs e) =>
            {
                Console.WriteLine("检测到说话开始...");
                try
                {
                    var session = await asrService.StartSessionAsync();
                    lock (_sessionLock)
                    {
                        _currentSessionId = session.SessionId;
                    }
                    Console.WriteLine($"ASR会话已启动: {session.SessionId}");

                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await foreach (var result in asrService.GetStreamResultAsync(session.SessionId, default))
                            {
                                if (!string.IsNullOrWhiteSpace(result.Text))
                                    Console.WriteLine($"[实时][{session.SessionId}] {result.Text}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[ASR错误][{session.SessionId}] {ex.Message}");
                        }
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"启动ASR会话失败: {ex.Message}");
                }
            };

            vadService.OnSpeaking += (object? sender, VADEventArgs e) =>
            {
                lock (_sessionLock)
                {
                    if (_currentSessionId != null)
                        asrService.FeedAudio(_currentSessionId, e.Data);
                }
            };

            vadService.OnSpeechEnded += async (object? sender, VADEventArgs e) =>
            {
                Console.WriteLine($"检测到说话结束，时长 {e.Duration?.TotalSeconds:F1}s");

                string? sessionId;
                lock (_sessionLock)
                {
                    sessionId = _currentSessionId;
                    _currentSessionId = null;
                }

                if (sessionId != null)
                {
                    try
                    {
                        var result = await asrService.FinishSessionAsync(sessionId);
                        Console.WriteLine($"\n=== 本轮识别结果 ===");
                        Console.WriteLine($"文本: {result?.Text ?? "(空)"}");
                        Console.WriteLine($"====================\n");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"结束ASR会话失败: {ex.Message}");
                    }
                }
            };

            recordService.OnDataAvailable += (e) =>
            {
                Span<float> denoiseBuffer = e.Length <= 4096
                    ? stackalloc float[e.Length]
                    : new float[e.Length];
                e.CopyTo(denoiseBuffer);

                int produced = dnrService.Process(denoiseBuffer);

                vadService.Predict(denoiseBuffer[..produced]);
            };

            await recordService.StartRecordingAsync();
            Console.WriteLine("开始录音... 按Q键停止录音。");

            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Escape || key.Key == ConsoleKey.Q)
                {
                    Console.WriteLine("正在停止录音...");
                    await recordService.StopRecordingAsync();
                    break;
                }
            }
        }
    }
}
