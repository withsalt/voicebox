using Microsoft.Extensions.Logging;
using VoiceBox.AEC.DTLN;
using VoiceBox.AEC.DTLN.Models;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService.Models;

namespace DTLNAecDemo
{
    /// <summary>
    /// DTLN-aec 实时 demo：
    /// 远端测试音(播放参考) + 麦克风采集 -> DTLN-aec -> 统计/可选监听。
    ///
    /// 默认不把麦克风回放到扬声器，避免诊断链路自己形成正反馈；按 M 可开启处理后麦克风监听。
    /// </summary>
    internal static class Program
    {
        private const int SampleRate = 16000;
        private const int ChannelCount = 1;

        private static volatile bool _aecEnabled = true;
        private static volatile bool _testToneEnabled = true;
        private static volatile bool _monitorMic;
        private static long _framesIn;
        private static long _xrunCount;
        private static IAcousticProcessingMetricsSource? _metricsSource;

        static async Task Main()
        {
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.SetMinimumLevel(LogLevel.Warning);
                builder.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var audio = new AudioServiceManager()
                .AddLogger(loggerFactory)
                .AddDtlnAecService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.ModelSize = DtlnAecModelSize.Units128;
                    o.ThreadsNum = 1;
                    o.ReferenceDelaySamples = 0;
                    o.OutputPrerollSamples = DtlnAecOptions.HopLength;
                    o.WarmupHops = 3;
                })
                .AddPortAudioDuplexService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.SuggestedLatency = 0.020;
                    o.MaxFrameSize = 256;
                    o.RecordBufferCapacity = 10;
                    o.PlayMaxQueueSize = 20;
                    o.EnableWasapiExclusiveMode = false;
                    o.EnableAec = true;
                    o.EnableDnr = false;
                    o.EnableHpf = false;
                })
                .Build();

            using var recordService = audio.GetAudioRecordInstance();
            using var playService = audio.GetAudioPlayInstance();
            _metricsSource = audio.GetAcousticProcessingMetricsSource();

            recordService.OnError += (_, e) => Console.WriteLine($"[录音错误] {e}");
            playService.OnError += (_, e) => Console.WriteLine($"[播放错误] {e}");
            playService.OnQueueOverflow += (_, e) => Console.WriteLine($"[播放队列溢出] 丢弃 {e}");
            recordService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };
            playService.OnXrun += (_, e) => { Interlocked.Increment(ref _xrunCount); Console.WriteLine($"[Xrun] {e}"); };

            recordService.OnDataAvailable += data =>
            {
                Interlocked.Add(ref _framesIn, data.Length);
                if (_monitorMic)
                    playService.TryEnqueue(data, out _);
            };

            PrintDevices(recordService);

            if (recordService is not PortAudioDuplexService fullDuplexService)
            {
                Console.WriteLine("[警告] 当前录音服务不是 PortAudioDuplexService，DTLN AEC 需要全双工 render 参考。");
                return;
            }

            fullDuplexService.EnableAec = _aecEnabled;
            await fullDuplexService.StartAsync();

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
                        _aecEnabled = !_aecEnabled;
                        fullDuplexService.EnableAec = _aecEnabled;
                        Console.WriteLine($"DTLN AEC: {(_aecEnabled ? "开" : "关")}");
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
            if (_metricsSource != null && _metricsSource.TryGetLatestMetrics(out var metrics))
            {
                string erle = metrics.HasEchoReturnLossEnhancement ? $"{metrics.EchoReturnLossEnhancement:F2} dB" : "-";
                string erl = metrics.HasEchoReturnLoss ? $"{metrics.EchoReturnLoss:F2} dB" : "-";
                string rel = metrics.HasResidualEchoLikelihood ? $"{metrics.ResidualEchoLikelihood:F2}" : "-";
                Console.WriteLine($"[统计] 已采集 {seconds:F1}s | 估算 ERLE {erle} | ERL {erl} | 残余相关 {rel} | Xrun {_xrunCount}");
            }
            else
            {
                Console.WriteLine($"[统计] 已采集 {seconds:F1}s | DTLN 指标暂不可用 | Xrun {_xrunCount}");
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
            Console.WriteLine("=== DTLN-aec 实时回声消除 Demo (16 kHz mono，全双工低延迟) ===");
            Console.WriteLine("  模型: dtln_aec_128，帧大小: 256 samples，内部 hop: 128 samples");
            Console.WriteLine($"  DTLN AEC: {(_aecEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine($"  远端测试音: {(_testToneEnabled ? "开" : "关")} (默认开)");
            Console.WriteLine($"  麦克风监听: {(_monitorMic ? "开" : "关")} (默认关)");
            Console.WriteLine("  A     - AEC 开/关 (A/B 对比)");
            Console.WriteLine("  T     - 远端测试音开/关");
            Console.WriteLine("  M     - 处理后麦克风监听开/关 (诊断用，易回授)");
            Console.WriteLine("  S     - 显示估算统计");
            Console.WriteLine("  Q/Esc - 退出");
            Console.WriteLine("  提示: 真实 TTS 链路里 render 必须来自实际播放队列；默认测试音只用于验证参考链路和双讲效果。");
            Console.WriteLine();
        }
    }
}
