using Microsoft.Extensions.Logging;
using VoiceBox.ASR.DoubaoOnline;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService.Models;
using VoiceBox.DNR.RNNoise;
using VoiceBox.VAD.FireRed;

namespace DoubaoOnlineAsrDemo
{
    /// <summary>
    /// 基于火山引擎「豆包流式语音识别模型 2.0」双向流式 WebSocket 接口的实时识别 Demo。
    /// 麦克风采集 → RNNoise 降噪 → FireRed VAD 分句 → 每段语音开一条 ASR 会话，实时上屏 + 收尾出终稿。
    ///
    /// 鉴权 (新版控制台) 通过环境变量提供：
    ///   VOLC_ASR_API_KEY      —— X-Api-Key (新版控制台 API Key)
    ///   VOLC_ASR_RESOURCE_ID  —— 可选，默认 volc.seedasr.sauc.duration (豆包 2.0 小时版)
    /// </summary>
    internal class Program
    {
        private static string? _currentSessionId;
        private static readonly object _sessionLock = new();

        /// <summary>
        /// 连接建立期间的音频预缓冲区。
        /// null  = 无正在建立中的会话（静默期或会话已就绪）。
        /// !null = StartSessionAsync 尚未完成，OnSpeaking 把帧存入此处，握手完成后整批回放。
        /// 全程由 _sessionLock 保护。
        /// </summary>
        private static List<float[]>? _preBuffer;

        static async Task Main(string[] args)
        {
            var (apiKey, resourceId) = ReadCredentials();
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                Console.WriteLine("未检测到鉴权配置。请设置环境变量后重试：");
                Console.WriteLine("  新版控制台: VOLC_ASR_API_KEY[, VOLC_ASR_RESOURCE_ID]");
                return;
            }

            using var audioServiceProxy = new AudioServiceManager()
                .AddLogger(b =>
                {
                    b.SetMinimumLevel(LogLevel.Information);
                    b.AddSimpleConsole(o =>
                    {
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
                })
                .AddRNNoiseDNRService(options =>
                {
                    options.SampleRate = 16000;
                    options.ChannelCount = 1;
                    options.SampleFormat = SampleFormat.Float32;
                })
                .AddDoubaoOnlineASRService(options =>
                {
                    options.ApiKey = apiKey;
                    if (!string.IsNullOrWhiteSpace(resourceId))
                        options.ResourceId = resourceId!;

                    options.StreamMode = SeedAsrStreamMode.Bidirectional;
                    options.PackageDurationMs = 100;
                    options.ShowUtterances = true;     // 便于区分已确定/未确定文本
                    options.EnablePunc = true;
                    options.EnableItn = true;
                    // 实时麦克风音频天然按实时到达，无需再节流。
                    options.ThrottleSendToRealtime = false;
                })
                .Build();

            using var recordService = audioServiceProxy.GetAudioRecordInstance();
            using var vadService = audioServiceProxy.GetVoiceActivityDetectInstance();
            using var dnrService = audioServiceProxy.GetDNRServiceInstance();
            using var asrService = audioServiceProxy.GetStreamASRServiceInstance();

            PrintInputDevices(recordService);

            WireVadHandlers(vadService, asrService);
            WireRecording(recordService, dnrService, vadService);

            await recordService.StartRecordingAsync();
            Console.WriteLine("开始录音... 说话即可实时识别，按 Q 或 Esc 停止。");

            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key is ConsoleKey.Escape or ConsoleKey.Q)
                {
                    Console.WriteLine("正在停止录音...");
                    await recordService.StopRecordingAsync();
                    break;
                }
            }
        }

        private static (string? apiKey, string? resourceId) ReadCredentials()
        {
            string? Get(string k) => Environment.GetEnvironmentVariable(k) is { Length: > 0 } v ? v : null;
            return (Get("VOLC_ASR_API_KEY"), Get("VOLC_ASR_RESOURCE_ID"));
        }

        private static void PrintInputDevices(IAudioRecordService recordService)
        {
            var inputDevices = recordService.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(s => s.Id).ToList();
            Console.WriteLine("可用输入设备:");
            foreach (var item in inputDevices ?? new())
                Console.WriteLine($"{item.Id}. {item.Name}{(item.IsDefault ? "【默认设备】" : "")}");
            Console.WriteLine();
        }

        private static void WireVadHandlers(
            VoiceBox.Core.Services.VadService.IVoiceActivityDetectService vadService,
            IStreamASRService asrService)
        {
            vadService.OnSpeechStarted += async (_, _) =>
            {
                Console.WriteLine("检测到说话开始...");

                // ① 立即开辟预缓冲区 —— OnSpeaking 在 WebSocket 握手期间把音频存入此处而非丢弃。
                lock (_sessionLock) { _preBuffer = new List<float[]>(); }

                try
                {
                    var session = await asrService.StartSessionAsync();

                    // ② 握手完成后：在同一把锁内原子地设置 SessionId、回放缓冲、清空缓冲区。
                    //    锁内 FeedAudio 均为 O(1) (Channel TryWrite)，不会长时间持锁。
                    lock (_sessionLock)
                    {
                        _currentSessionId = session.SessionId;
                        if (_preBuffer is { Count: > 0 } buf)
                        {
                            Console.WriteLine($"回放 {buf.Count} 帧连接期间缓冲音频...");
                            foreach (var chunk in buf)
                                asrService.FeedAudio(session.SessionId, chunk);
                        }
                        _preBuffer = null;
                    }

                    Console.WriteLine($"ASR 会话已启动: {session.SessionId}");

                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await foreach (var result in asrService.GetStreamResultAsync(session.SessionId))
                            {
                                if (!string.IsNullOrWhiteSpace(result.Text))
                                    Console.WriteLine($"[实时] {result.Text}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[ASR错误] {ex.Message}");
                        }
                    });
                }
                catch (Exception ex)
                {
                    // 连接失败时清空缓冲区，避免内存泄漏。
                    lock (_sessionLock) { _preBuffer = null; }
                    Console.WriteLine($"启动 ASR 会话失败: {ex.Message}");
                }
            };

            vadService.OnSpeaking += (_, e) =>
            {
                lock (_sessionLock)
                {
                    if (_currentSessionId != null)
                    {
                        // ③ 会话就绪：直接喂入。
                        asrService.FeedAudio(_currentSessionId, e.Data);
                    }
                    else if (_preBuffer != null)
                    {
                        // ④ 连接建立中：把音频复制一份存入预缓冲区 (e.Data 是 span，不能跨 await 持有)。
                        _preBuffer.Add(e.Data.ToArray());
                    }
                }
            };

            vadService.OnSpeechEnded += async (_, e) =>
            {
                Console.WriteLine($"检测到说话结束，时长 {e.Duration?.TotalSeconds:F1}s");
                string? sessionId;
                lock (_sessionLock)
                {
                    sessionId = _currentSessionId;
                    _currentSessionId = null;
                    _preBuffer = null; // 兜底: 若 StartSession 尚未完成就触发结束，清掉缓冲。
                }
                if (sessionId is null) return;

                try
                {
                    var result = await asrService.FinishSessionAsync(sessionId);
                    Console.WriteLine("\n=== 本轮识别结果 ===");
                    Console.WriteLine($"文本: {result?.Text ?? "(空)"}");
                    Console.WriteLine("====================\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"结束 ASR 会话失败: {ex.Message}");
                }
            };
        }

        private static void WireRecording(
            IAudioRecordService recordService,
            VoiceBox.Core.Services.DNRService.IDigitalNoiseReductionService dnrService,
            VoiceBox.Core.Services.VadService.IVoiceActivityDetectService vadService)
        {
            recordService.OnDataAvailable += (e) =>
            {
                Span<float> buffer = e.Length <= 4096 ? stackalloc float[e.Length] : new float[e.Length];
                e.CopyTo(buffer);
                int produced = dnrService.Process(buffer);
                vadService.Predict(buffer[..produced]);
            };
        }
    }
}
