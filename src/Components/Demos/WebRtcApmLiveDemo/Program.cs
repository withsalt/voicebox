using Microsoft.Extensions.Logging;
using VoiceBox.APM.WebRtc;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;

namespace WebRtcApmLiveDemo
{
    /// <summary>
    /// WebRTC M151 统一采集侧 APM 处理器实时 demo：
    /// 扬声器测试音(远端参考) + 麦克风采集 → 单实例 APM(HPF + AEC3 + NS) → 统计。
    ///
    /// 全双工低延迟模式下，播放(远端)信号作为 AEC 的对齐参考，由 <see cref="WebRtcApmService"/>
    /// 在回调内一次性完成全部采集侧处理，取代分离的 AEC + DNR。
    ///
    /// 默认播放独立测试音作为 AEC 远端参考，不把麦克风回放到扬声器。旧版"麦克风→扬声器"
    /// 自监听回环容易形成正反馈/梳状滤波，听到断续嗡嗡声并不代表产品 TTS 链路。
    ///
    /// 操作：
    ///   A 键 — 切换 APM 开/关，即时 A/B 对比 (外放时尤为明显)
    ///   T 键 — 切换远端测试音开/关
    ///   M 键 — 切换麦克风监听(诊断用，易回授)
    ///   S 键 — 显示 APM 实时统计 (ERLE / 延迟 / 残余回声似然)
    ///   Q/Esc— 退出
    ///
    /// 提示：关闭 APM 并外放时可能产生回授啸叫；开启后应被 AEC3 抑制。
    /// </summary>
    internal static class Program
    {
        private const int SampleRate = 16000;
        private const int ChannelCount = 1;

        private static volatile bool _apmEnabled = true;
        private static volatile bool _testToneEnabled = true;
        private static volatile bool _monitorMic;
        private static long _framesIn;
        private static long _xrunCount;
        private static WebRtcApmService? _apmService;

        static async Task Main()
        {
            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Warning);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var audio = new AudioServiceManager()
                .AddWebRtcApmService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;

                    // 统一处理器：HPF + AEC3 + NS。
                    o.EnableHighPassFilter = true;
                    o.EnableEchoCanceller = true;
                    o.EnableNoiseSuppression = true;

                    // 关键：关闭 AGC2。其自适应数字增益会在安静/低电平时猛拉噪声底与 AEC 残差，
                    // 在"麦克风→扬声器"回环里更会形成正反馈 → 断续嗡嗡声/啸叫。
                    o.EnableAgc2 = false;

                    // AEC3 双讲调参(默认值；调小 DominantNearendEnrThreshold 更倾向保留近端语音)。
                    //o.DominantNearendEnrThreshold = 0.1f;
                })
                .AddPortAudioDuplexService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;

                    // 低延迟配置
                    o.SuggestedLatency = 0.020;   // 20ms
                    o.MaxFrameSize = 256;          // 16ms @ 16kHz
                    o.RecordBufferCapacity = 10;
                    o.PlayMaxQueueSize = 20;

                    o.EnableWasapiExclusiveMode = false;

                    // 关键：统一 APM 挂在 AEC 槽位；独立 HPF/DNR 一律关闭(已由 APM 一并完成)。
                    o.EnableAec = true;
                    o.EnableDnr = false;
                    o.EnableHpf = false;
                })
                .Build();

            using var recordService = audio.GetAudioRecordInstance();
            using var playService = audio.GetAudioPlayInstance();
            _apmService = audio.GetAECServiceInstance() as WebRtcApmService;

            recordService.OnError += (_, e) => Console.WriteLine($"[录音错误] {e}");
            playService.OnError += (_, e) => Console.WriteLine($"[播放错误] {e}");
            playService.OnQueueOverflow += (_, e) => Console.WriteLine($"[播放队列溢出] 丢弃 {e}");
            recordService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };
            playService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };

            // 核心实时链路：录音块已由 PortAudioDuplexService 完成 APM 处理。
            // 默认不回放麦克风，避免自监听正反馈；需要听处理后麦克风时按 M 开启诊断监听。
            recordService.OnDataAvailable += (data) =>
            {
                Interlocked.Add(ref _framesIn, data.Length);
                if (_monitorMic)
                    playService.TryEnqueue(data, out _);
            };

            PrintDevices(recordService);

            var fullDuplexService = recordService as PortAudioDuplexService;
            if (fullDuplexService != null)
            {
                fullDuplexService.EnableAec = _apmEnabled;
                await fullDuplexService.StartAsync();
            }
            else
            {
                Console.WriteLine("[警告] 非全双工模式，APM 回声消除不可用 (需要对齐的播放参考信号)。");
                return;
            }

            using var toneCts = new CancellationTokenSource();
            var toneTask = Task.Run(() => RunFarEndToneAsync(playService, toneCts.Token));

            PrintHelp();
            Console.WriteLine("正在实时处理... 默认播放远端测试音，不回放麦克风 (按 Q 退出)");

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
                        _apmEnabled = !_apmEnabled;
                        fullDuplexService.EnableAec = _apmEnabled;
                        Console.WriteLine($"统一 APM (HPF+AEC3+NS): {(_apmEnabled ? "开" : "关")}");
                        break;

                    case ConsoleKey.T:
                        _testToneEnabled = !_testToneEnabled;
                        Console.WriteLine($"远端测试音: {(_testToneEnabled ? "开" : "关")}");
                        break;

                    case ConsoleKey.M:
                        _monitorMic = !_monitorMic;
                        if (_monitorMic)
                            _testToneEnabled = false;
                        Console.WriteLine($"麦克风监听: {(_monitorMic ? "开 (诊断用，注意音量)" : "关")}");
                        break;

                    case ConsoleKey.S:
                        PrintStats();
                        break;
                }
            }

            Console.WriteLine("正在停止...");
            toneCts.Cancel();
            try { await toneTask.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            await fullDuplexService.StopAsync();
            Console.WriteLine($"已退出。总 Xrun 次数: {_xrunCount}");
        }

        private static async Task RunFarEndToneAsync(VoiceBox.Core.Services.AudioService.IAudioPlayService playService, CancellationToken ct)
        {
            const int chunkMs = 20;
            const int samplesPerChunk = SampleRate * chunkMs / 1000;
            var buffer = new float[samplesPerChunk];
            double phase = 0;
            long sampleClock = 0;

            while (!ct.IsCancellationRequested)
            {
                if (_testToneEnabled && !_monitorMic && playService.CurrentQueueSize < 8)
                {
                    FillTestTone(buffer, ref phase, sampleClock);
                    sampleClock += buffer.Length;
                    playService.TryEnqueue(buffer, out _);
                }

                await Task.Delay(chunkMs, ct).ConfigureAwait(false);
            }
        }

        private static void FillTestTone(float[] buffer, ref double phase, long sampleClock)
        {
            // 用两个频率交替的低电平测试音，避免单一纯音让 AEC 统计过于乐观。
            double seconds = sampleClock / (double)SampleRate;
            double freq = (seconds % 1.6) < 0.8 ? 440.0 : 880.0;
            double step = 2.0 * Math.PI * freq / SampleRate;
            for (int i = 0; i < buffer.Length; i++)
            {
                double envelope = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * (i + 1) / buffer.Length);
                buffer[i] = (float)(0.12 * envelope * Math.Sin(phase));
                phase += step;
                if (phase > 2.0 * Math.PI)
                    phase -= 2.0 * Math.PI;
            }
        }

        private static void PrintStats()
        {
            double seconds = _framesIn / (double)SampleRate;
            if (_apmService != null && _apmService.TryGetStats(out var s))
            {
                string erle = s.HasEchoReturnLossEnhancement ? $"{s.EchoReturnLossEnhancement:F2} dB" : "—";
                string erl = s.HasEchoReturnLoss ? $"{s.EchoReturnLoss:F2} dB" : "—";
                string delay = s.HasDelayMs ? $"{s.DelayMs} ms" : "—";
                string rel = s.HasResidualEchoLikelihood ? $"{s.ResidualEchoLikelihood:F2}" : "—";
                Console.WriteLine($"[统计] 已采集 {seconds:F1}s | ERLE {erle} | ERL {erl} | 延迟 {delay} | 残余回声似然 {rel} | Xrun {_xrunCount}");
            }
            else
            {
                Console.WriteLine($"[统计] 已采集 {seconds:F1}s | APM 统计不可用 | Xrun {_xrunCount}");
            }
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
            Console.WriteLine("=== WebRTC M151 统一 APM 处理器 (HPF + AEC3 + NS，全双工低延迟) ===");
            Console.WriteLine($"  延迟配置: 20ms, 帧大小: 256 samples");
            Console.WriteLine($"  统一 APM: {(_apmEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine($"  远端测试音: {(_testToneEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine($"  麦克风监听: {(_monitorMic ? "开" : "关")} (默认关)");
            Console.WriteLine("  A     - APM 开/关 (A/B 对比)");
            Console.WriteLine("  T     - 远端测试音开/关");
            Console.WriteLine("  M     - 麦克风监听开/关 (诊断用，易回授)");
            Console.WriteLine("  S     - 显示 APM 实时统计 (ERLE/延迟/残余回声)");
            Console.WriteLine("  Q/Esc - 退出");
            Console.WriteLine("  提示: 默认测试音用于验证 AEC 参考链路；麦克风监听只用于听感诊断。");
            Console.WriteLine();
        }
    }
}
