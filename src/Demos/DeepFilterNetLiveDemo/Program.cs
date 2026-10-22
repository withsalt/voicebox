using Microsoft.Extensions.Logging;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DeepFilterNet;
using VoiceBox.DNR.DeepFilterNet.Models;

namespace DeepFilterNetLiveDemo
{
    /// <summary>
    /// DeepFilterNet (DF3-LL) 实时流式降噪 demo：麦克风采集 → DeepFilterNet 逐帧降噪 → 扬声器播放。
    ///
    /// DeepFilterNet 原生工作于 48 kHz，故全双工管线直接采用 48 kHz、无重采样以获得最佳音质与最低延迟。
    /// 整条链路按音频块流式处理 (Denoise(..., finish:false))，跨块保留模型三个 GRU 状态与
    /// STFT/ISTFT 缓冲，演示真·逐帧 (~10ms) 低延迟降噪。DeepFilterNet 通过 DI 注入到
    /// PortAudioDuplexService，由全双工服务在录音派发前完成降噪。
    ///
    /// 操作：
    ///   D 键 — 切换降噪开/关 (旁路)，可即时 A/B 对比
    ///   S 键 — 显示统计
    ///   Q/Esc— 退出
    /// </summary>
    internal static class Program
    {
        private const int SampleRate = 48000;
        private const int ChannelCount = 1;

        private static volatile bool _denoiseEnabled = true;
        private static long _framesIn;
        private static long _xrunCount;

        static async Task Main()
        {
            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Warning);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var audio = new AudioServiceManager()
                .AddDeepFilterNetDNRService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.Model = DeepFilterNetModel.Dfn3LowLatency;
                })
                .AddPortAudioDuplexService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;

                    o.SuggestedLatency = 0.020;    // 20ms
                    o.MaxFrameSize = 480;          // 10ms @ 48kHz（与 DeepFilterNet hop 对齐）
                    o.RecordBufferCapacity = 10;
                    o.PlayMaxQueueSize = 20;
                    o.EnableWasapiExclusiveMode = false;
                    o.EnableAec = false;
                    o.EnableDnr = true;
                })
                .Build();

            using var recordService = audio.GetAudioRecordInstance();
            using var playService = audio.GetAudioPlayInstance();

            recordService.OnError += (_, e) => Console.WriteLine($"[录音错误] {e}");
            playService.OnError += (_, e) => Console.WriteLine($"[播放错误] {e}");
            playService.OnQueueOverflow += (_, e) => Console.WriteLine($"[播放队列溢出] 丢弃 {e}");
            recordService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };
            playService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };

            // 核心实时链路：录音块已由 PortAudioDuplexService 完成 DNR → 入播放队列。
            recordService.OnDataAvailable += (data) =>
            {
                Interlocked.Add(ref _framesIn, data.Length);
                playService.TryEnqueue(data, out _);
            };

            PrintDevices(recordService);

            var fullDuplexService = recordService as PortAudioDuplexService;
            if (fullDuplexService != null)
            {
                fullDuplexService.EnableDnr = _denoiseEnabled;
                await fullDuplexService.StartAsync();
            }
            else
            {
                Console.WriteLine("[警告] 非全双工模式，回退到分离录放。");
                await playService.StartPlayingAsync();
                await recordService.StartRecordingAsync();
            }

            PrintHelp();
            Console.WriteLine("正在实时降噪监听... (按 Q 退出)");

            bool running = true;
            while (running)
            {
                var key = Console.ReadKey(true);
                switch (key.Key)
                {
                    case ConsoleKey.Q:
                    case ConsoleKey.Escape:
                        running = false;
                        break;

                    case ConsoleKey.D:
                        _denoiseEnabled = !_denoiseEnabled;
                        if (fullDuplexService != null)
                            fullDuplexService.EnableDnr = _denoiseEnabled;
                        Console.WriteLine($"降噪: {(_denoiseEnabled ? "开" : "关 (旁路)")}");
                        break;

                    case ConsoleKey.S:
                        Console.WriteLine($"已采集 {_framesIn / (double)SampleRate:F1}s，" +
                                          $"播放队列 {playService.CurrentQueueSize}，" +
                                          $"Xrun 次数 {_xrunCount}");
                        break;
                }
            }

            Console.WriteLine("正在停止...");
            if (fullDuplexService != null)
            {
                await fullDuplexService.StopAsync();
            }
            else
            {
                await recordService.StopRecordingAsync();
                await playService.StopPlayingAsync();
            }
            Console.WriteLine($"已退出。总 Xrun 次数: {_xrunCount}");
        }

        private static void PrintDevices(VoiceBox.Core.Services.AudioService.IAudioRecordService rec)
        {
            var inputs = rec.ListAudioDevices(AudioDeviceType.Input)?.OrderBy(s => s.Id);
            Console.WriteLine("可用输入设备:");
            foreach (var d in inputs ?? Enumerable.Empty<AudioDevice>())
                Console.WriteLine($"  {d.Id}. {d.Name}{(d.IsDefault ? " 【默认】" : "")}");
            Console.WriteLine();
        }

        private static void PrintHelp()
        {
            Console.WriteLine();
            Console.WriteLine("=== DeepFilterNet (DF3-LL) 实时降噪（全双工 48kHz 低延迟）===");
            Console.WriteLine($"  采样率: {SampleRate}Hz, 帧大小: 480 samples (10ms), 算法延迟≈10ms");
            Console.WriteLine($"  降噪(DNR): {(_denoiseEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine("  D     - 降噪开/关 (A/B 对比)");
            Console.WriteLine("  S     - 显示统计");
            Console.WriteLine("  Q/Esc - 退出");
            Console.WriteLine();
        }
    }
}
