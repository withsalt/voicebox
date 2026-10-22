using Microsoft.Extensions.Logging;
using VoiceBox.AEC.DFSMN;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.DNR.DPDFNet.Models;
using VoiceBox.DNR.RNNoise;

namespace DPDFNetLiveDemo
{
    /// <summary>
    /// DPDFNet 实时流式降噪 demo：麦克风采集 → DPDFNet 流式降噪 → 扬声器播放。
    ///
    /// 整条链路按音频块流式处理 (Denoise(..., finish: false))，跨块保留模型循环状态与
    /// STFT/ISTFT、重采样缓冲，演示在实时监听场景下的低延迟降噪。
    ///
    /// 使用全双工模式。DPDFNet 通过 DI 注入到
    /// PortAudioDuplexService，由全双工服务在录音派发前完成降噪。
    ///
    /// 操作：
    ///   A 键 — 切换回声消除开/关，可即时 A/B 对比 (外放时尤为明显)
    ///   D 键 — 切换降噪开/关 (旁路)，可即时 A/B 对比
    ///   Q/Esc— 退出
    ///
    /// 提示：关闭回声消除并外放时可能产生回授啸叫；开启后应被抑制。
    /// </summary>
    internal static class Program
    {
        private const int SampleRate = 16000;
        private const int ChannelCount = 1;

        private static volatile bool _denoiseEnabled = true;
        private static volatile bool _aecEnabled = true;
        private static long _framesIn;
        private static long _framesOut;
        private static long _xrunCount;

        static async Task Main()
        {
            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Warning);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            // 使用全双工模式
            using var audio = new AudioServiceManager()
                .AddDfsmnAecService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                })
                .AddDPDFNetDNRService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.Model = DPDFNetModel.DpDfNet4;
                    o.AttenuationLimitDb = 12f;  // 限制 DNR 单帧最大增益 6dB，防止放大 AEC 残留尾音
                })
                .AddPortAudioDuplexService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;

                    o.SuggestedLatency = 0.020;   // 20ms 延迟
                    o.MaxFrameSize = 256;          // 16ms @ 16kHz
                    o.RecordBufferCapacity = 10;
                    o.PlayMaxQueueSize = 20;

                    // Windows WASAPI 独占模式（可选）
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

            // 核心实时链路：录音块已由 PortAudioDuplexService 完成 AEC/DNR → 入播放队列。
            recordService.OnDataAvailable += (data) =>
            {
                Interlocked.Add(ref _framesIn, data.Length);
                Interlocked.Add(ref _framesOut, data.Length);
                playService.TryEnqueue(data, out _);
            };

            PrintDevices(recordService);

            // 全双工模式：同时启动录音和播放
            // 使用 PortAudioFullDuplexService 的 StartAsync 方法
            var fullDuplexService = recordService as PortAudioDuplexService;
            if (fullDuplexService != null)
            {
                fullDuplexService.EnableAec = _aecEnabled;
                fullDuplexService.EnableDnr = _denoiseEnabled;
                await fullDuplexService.StartAsync();
            }
            else
            {
                Console.WriteLine("[警告] 非全双工模式，回声消除不可用 (AEC 需要对齐的播放参考信号)。");
                // 回退到分离模式
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

                    case ConsoleKey.A:
                        _aecEnabled = !_aecEnabled;
                        if (fullDuplexService != null)
                            fullDuplexService.EnableAec = _aecEnabled;
                        Console.WriteLine($"回声消除(AEC): {(_aecEnabled ? "开" : "关")}");
                        break;

                    case ConsoleKey.D:
                        _denoiseEnabled = !_denoiseEnabled;
                        if (fullDuplexService != null)
                            fullDuplexService.EnableDnr = _denoiseEnabled;
                        Console.WriteLine($"降噪: {(_denoiseEnabled ? "开" : "关 (旁路)")}");
                        break;

                    case ConsoleKey.S:
                        Console.WriteLine($"已采集 {_framesIn / (double)SampleRate:F1}s，" +
                                          $"输出 {_framesOut / (double)SampleRate:F1}s，" +
                                          $"播放队列 {playService.CurrentQueueSize}，" +
                                          $"Xrun 次数 {_xrunCount}");
                        break;
                }
            }

            Console.WriteLine("正在停止...");
            var fullDuplexServiceStop = recordService as VoiceBox.Audio.PortAudio.PortAudioDuplexService;
            if (fullDuplexServiceStop != null)
            {
                await fullDuplexServiceStop.StopAsync();
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
            Console.WriteLine("=== DPDFNet 实时降噪 + WebRTC AEC 回声消除（全双工低延迟模式）===");
            Console.WriteLine($"  延迟配置: 20ms, 帧大小: 256 samples");
            Console.WriteLine($"  回声消除(AEC): {(_aecEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine($"  降噪(DNR): {(_denoiseEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine("  A     - 回声消除开/关 (A/B 对比)");
            Console.WriteLine("  D     - 降噪开/关 (A/B 对比)");
            Console.WriteLine("  S     - 显示统计");
            Console.WriteLine("  Q/Esc - 退出");
            Console.WriteLine("  提示: 关 AEC 外放易回授啸叫；开 AEC 后可外放对比效果。");
            Console.WriteLine();
        }
    }
}
