using System.Buffers;
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Speaker.WeSpeaker;

namespace WeSpeakerSpeakerLiveDemo
{
    /// <summary>
    /// WeSpeaker 实时说话人检测 demo: 麦克风采集 → 滑动窗口声纹比对 (0.5s 窗 / 0.2s 跳)。
    ///
    /// 流程: 按 E 录制 5 秒连续音频注册当前说话人, 之后持续显示实时相似度得分条;
    /// 所有采集音频都直接送入 WeSpeaker, 推理在独立工作线程执行, 避免阻塞采集。
    ///
    /// 操作:
    ///   E 键 — 录制 5s 并注册/覆盖说话人 "me"
    ///   R 键 — 重置流式状态 (清平滑得分)
    ///   Q/Esc— 退出
    /// </summary>
    internal static class Program
    {
        private const int SampleRate = 16000;
        private const float EnrollSeconds = 5f;
        private const float EnrollTimeoutSeconds = 15f;

        private static volatile bool _enrolling;
        private static readonly List<float> _enrollBuffer = new();
        private static readonly BlockingCollection<(float[] Buffer, int Length, bool ResetBefore)> _audioQueue = new(boundedCapacity: 64);
        private static readonly object _audioQueueLock = new();
        private static readonly object _consoleLock = new();
        private static bool _resetBeforeNextAudio = true;
        private static bool _liveStatusVisible;
        private static long _observedWindowCount;
        private static long _matchedWindowCount;

        static async Task Main()
        {
            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Warning);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss "; });
            });

            using var audio = new AudioServiceManager()
                .AddPortAudioRecordService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = 1;
                    o.SampleFormat = SampleFormat.Float32;
                })
                .AddWeSpeakerService(o =>
                {
                    o.SampleRate = SampleRate;
                    o.ChannelCount = 1;
                    o.SampleFormat = SampleFormat.Float32;

                    o.WindowMs = 500;   // 与非 LM 模型的 200 帧训练长度对齐
                    o.HopMs = 200;
                    o.ScoreSmoothing = 0.6f;
                    o.Threshold = 0.3f;
                    o.MinWindowRms = 0f;       // 不跳过静音窗，持续产出每个滑动窗口的结果
                })
                .Build();

            using var recordService = audio.GetAudioRecordInstance();
            using var sv = audio.GetSpeakerServiceInstance();

            sv.OnResult += (_, e) =>
            {
                if (e.BestSpeaker is null) return;
                long observedWindows = Interlocked.Increment(ref _observedWindowCount);
                long matchedWindows = e.IsMatch
                    ? Interlocked.Increment(ref _matchedWindowCount)
                    : Interlocked.Read(ref _matchedWindowCount);
                int barWidth = GetStrengthBarWidth(e.BestSpeaker);
                int bar = (int)Math.Clamp((e.BestScore + 0.2f) / 1.2f * barWidth, 0, barWidth);
                string strengthLine = $"|{new string('#', bar).PadRight(barWidth)}| " +
                                      $"speaker={e.BestSpeaker} {(e.IsMatch ? "本人" : "其他/不确定")}";
                string infoLine = $"t={e.AudioTime.TotalSeconds,7:F2}s score={e.BestScore,7:F4} " +
                                  $"threshold={sv.Threshold:F4} windows={matchedWindows}/{observedWindows}";
                WriteLiveStatus(strengthLine, infoLine);
            };

            recordService.OnDataAvailable += data =>
            {
                if (_enrolling)
                {
                    lock (_enrollBuffer)
                    {
                        for (int i = 0; i < data.Length; i++) _enrollBuffer.Add(data[i]);
                    }
                    return;
                }

                var buffer = ArrayPool<float>.Shared.Rent(data.Length);
                data.CopyTo(buffer);
                try
                {
                    lock (_audioQueueLock)
                    {
                        bool resetBefore = _resetBeforeNextAudio;
                        if (_audioQueue.TryAdd((buffer, data.Length, resetBefore)))
                        {
                            if (resetBefore) _resetBeforeNextAudio = false;
                        }
                        else // 队列满: 丢块保实时；重置信号保留给下一块
                        {
                            ArrayPool<float>.Shared.Return(buffer);
                        }
                    }
                }
                catch (InvalidOperationException) // CompleteAdding 之后的尾部回调
                {
                    ArrayPool<float>.Shared.Return(buffer);
                }
            };

            var worker = Task.Run(() =>
            {
                foreach (var (buffer, length, resetBefore) in _audioQueue.GetConsumingEnumerable())
                {
                    try
                    {
                        if (resetBefore) sv.Reset();
                        sv.FeedAudio(buffer.AsSpan(0, length));
                    }
                    catch (ObjectDisposedException) { /* 退出中 */ }
                    finally { ArrayPool<float>.Shared.Return(buffer); }
                }
            });

            await recordService.StartRecordingAsync();
            Console.WriteLine($"实时说话人检测 (WeSpeaker, 无 VAD, threshold={sv.Threshold:F4})。按 E 注册当前说话人 (采集 5s 连续音频), R 重置, Q 退出。");

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

                    case ConsoleKey.E:
                        await EnrollAsync(sv);
                        break;

                    case ConsoleKey.R:
                        sv.Reset();
                        lock (_audioQueueLock) _resetBeforeNextAudio = true;
                        ResetWindowCounts();
                        WriteConsoleMessage("已重置 WeSpeaker 流式状态和窗口计数。");
                        break;
                }
            }

            // 先停采集再封队列, 避免尾部回调撞上 CompleteAdding
            await recordService.StopRecordingAsync();
            _audioQueue.CompleteAdding();
            await worker;
        }

        private static async Task EnrollAsync(VoiceBox.Core.Services.SpeakerService.ISpeakerService sv)
        {
            lock (_enrollBuffer) _enrollBuffer.Clear();
            _enrolling = true;
            WriteConsoleMessage(
                "开始注册:",
                "请自然朗读：清晨的阳光照进窗台，我端起一杯温水，准备开始今天的工作。请记住我的声音，我们稍后再见。",
                $"请连续朗读 {EnrollSeconds:F0} 秒（最多等待 {EnrollTimeoutSeconds:F0} 秒采集完成）。");

            int needed = (int)(SampleRate * EnrollSeconds);
            DateTime deadline = DateTime.UtcNow.AddSeconds(EnrollTimeoutSeconds);
            while (true)
            {
                await Task.Delay(100);
                lock (_enrollBuffer)
                {
                    if (_enrollBuffer.Count >= needed) break;
                }
                if (DateTime.UtcNow >= deadline)
                {
                    int collected;
                    lock (_enrollBuffer) collected = _enrollBuffer.Count;
                    WriteConsoleMessage($"注册超时：仅采集到 {collected / (float)SampleRate:F1}s 音频，已取消。");
                    _enrolling = false;
                    return;
                }
            }

            float[] samples;
            lock (_enrollBuffer) samples = _enrollBuffer.ToArray();
            double rms = Math.Sqrt(samples.Average(s => (double)s * s));
            if (rms < 0.005)
            {
                WriteConsoleMessage($"注册音频几乎是静音 (RMS={rms:F4}), 已取消。请靠近麦克风重试。");
                _enrolling = false;
                return;
            }

            sv.Enroll("me", samples);
            sv.Reset();
            lock (_audioQueueLock) _resetBeforeNextAudio = true;
            ResetWindowCounts();
            WriteConsoleMessage($"注册完成 (连续音频 {samples.Length / (float)SampleRate:F1}s, RMS={rms:F4})。开始实时检测, 请继续说话观察得分和窗口命中数。");
            _enrolling = false;
        }

        private static void ResetWindowCounts()
        {
            Interlocked.Exchange(ref _observedWindowCount, 0);
            Interlocked.Exchange(ref _matchedWindowCount, 0);
        }

        private static int GetStrengthBarWidth(string speaker)
        {
            try
            {
                return Math.Clamp(Console.WindowWidth - speaker.Length - 28, 10, 40);
            }
            catch (IOException)
            {
                return 40;
            }
        }

        private static void WriteLiveStatus(string strengthLine, string infoLine)
        {
            lock (_consoleLock)
            {
                try
                {
                    if (_liveStatusVisible && Console.CursorTop > 0)
                        Console.SetCursorPosition(0, Console.CursorTop - 1);

                    ClearCurrentConsoleLine();
                    Console.WriteLine(strengthLine);
                    ClearCurrentConsoleLine();
                    Console.Write(infoLine);
                    _liveStatusVisible = true;
                }
                catch (Exception ex) when (ex is IOException or ArgumentOutOfRangeException)
                {
                    Console.WriteLine(strengthLine);
                    Console.WriteLine(infoLine);
                    _liveStatusVisible = false;
                }
            }
        }

        private static void WriteConsoleMessage(params string[] lines)
        {
            lock (_consoleLock)
            {
                if (_liveStatusVisible)
                {
                    Console.WriteLine();
                    _liveStatusVisible = false;
                }

                foreach (string line in lines)
                    Console.WriteLine(line);
            }
        }

        private static void ClearCurrentConsoleLine()
        {
            int row = Console.CursorTop;
            int clearWidth = Math.Max(0, Console.BufferWidth - 1);
            Console.SetCursorPosition(0, row);
            Console.Write(new string(' ', clearWidth));
            Console.SetCursorPosition(0, row);
        }
    }
}
