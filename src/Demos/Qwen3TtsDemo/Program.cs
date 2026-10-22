using System.Globalization;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.Qwen3;

namespace Qwen3TtsDemo
{
    internal static class Program
    {
        // Qwen3-TTS 包目录。默认 fp32 (本机 RTX 5070 Ti 实测最快且与 PyTorch 贪心逐 token 字节一致)。
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "tts", "qwen3", "qwen3-tts-0.6b-onnx-fp32");

        // 执行提供者。默认 Cuda (实时性最佳); 设 QWEN3_PROVIDER=cpu 切 CPU。
        private static readonly OnnxExecutionProvider Provider = OnnxExecutionProvider.Cpu;


        private const int SampleRate = 24000;
        private const int ChannelCount = 1;

        static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (!Directory.Exists(ModelPath))
            {
                Console.WriteLine($"模型目录不存在: {ModelPath}");
                Console.WriteLine("请先运行: python scripts/qwen3-tts-onnx-export/export.py --dtype fp32");
                return;
            }

            // 无头基准模式: `Qwen3TtsDemo --bench "一句话"` (或直接把句子作为首个参数)。
            // 只测首包/RTF, 不启动 PortAudio。可用环境变量调参而无需重编译:
            //   QWEN3_MODEL=<包目录>  QWEN3_PROVIDER=cuda|cpu  QWEN3_CHUNKS=4,8,16,32,50
            //   QWEN3_VOICE=<音色>    QWEN3_ITERS=<重复次数>   QWEN3_TARGET_MS=<达标阈值>
            string? benchText = ExtractBenchText(args);
            if (benchText is not null)
            {
                await RunHeadlessBenchAsync(benchText);
                return;
            }

            await RunInteractiveAsync();
        }

        // ---------------------------------------------------------------------
        // 无头基准 (判定首包是否达标, 默认 <300ms)
        // ---------------------------------------------------------------------
        private static async Task RunHeadlessBenchAsync(string text)
        {
            int iters = TryGetEnvInt("QWEN3_ITERS") ?? 3;
            int targetMs = TryGetEnvInt("QWEN3_TARGET_MS") ?? 300;
            int[] schedule = ParseSchedule(Environment.GetEnvironmentVariable("QWEN3_CHUNKS"));

            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Warning);   // 安静些, 只看基准行
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss.fff "; });
            });

            var ttsOptions = BuildTtsOptions(schedule);
            using var tts = new Qwen3TTSService(Options.Create(ttsOptions),
                loggerFactory.CreateLogger<Qwen3TTSService>());

            // 触发模型加载 + 预热 (一次性, 不计入首包)。
            var loadSw = System.Diagnostics.Stopwatch.StartNew();
            var voices = tts.ListVoices();
            loadSw.Stop();
            string voice = Environment.GetEnvironmentVariable("QWEN3_VOICE") is { Length: > 0 } v
                ? v : (voices.Count > 0 ? voices[0].Name : "");

            Console.WriteLine("== Qwen3-TTS bench ==");
            Console.WriteLine($"   model    : {ModelPath}");
            Console.WriteLine($"   provider : {Provider}   voice: {voice}");
            Console.WriteLine($"   schedule : [{string.Join(",", ttsOptions.StreamChunkSchedule)}]   load+warmup: {loadSw.ElapsedMilliseconds}ms");
            Console.WriteLine($"   target   : 首包 < {targetMs}ms");
            Console.WriteLine($"   text     : {text}");
            Console.WriteLine();

            double bestFirst = double.MaxValue;
            bool whole = Environment.GetEnvironmentVariable("QWEN3_WHOLE") == "1";
            for (int i = 0; i < iters; i++)
            {
                if (whole)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    var r = await tts.GetAudioAsync(text, new TtsSynthesisOptions { Voice = voice, Language = "auto" }).ConfigureAwait(false);
                    sw.Stop();
                    double d = (double)r.Samples.Length / SampleRate;
                    Console.WriteLine($"[whole {i + 1}/{iters}] 时长={d,5:F2}s  墙钟={sw.Elapsed.TotalSeconds,5:F2}s  RTF={sw.Elapsed.TotalSeconds / Math.Max(1e-6, d):F3}");
                    continue;
                }
                var (firstMs, durSec, wallSec, chunks) = await MeasureStreamAsync(tts, text, voice);
                double rtf = wallSec / Math.Max(1e-6, durSec);
                bestFirst = Math.Min(bestFirst, firstMs);
                string verdict = firstMs <= targetMs ? "PASS" : "FAIL";
                Console.WriteLine($"[run {i + 1}/{iters}] 首包={firstMs,6:F0}ms  时长={durSec,5:F2}s  墙钟={wallSec,5:F2}s  RTF={rtf:F3}  块={chunks}  [{verdict}]");
            }
            if (whole) return;
            Console.WriteLine();
            Console.WriteLine($"== 最佳首包 = {bestFirst:F0}ms  ({(bestFirst <= targetMs ? "达标 PASS" : "未达标 FAIL")} @ {targetMs}ms) ==");
        }

        /// <summary>跑一次流式合成, 返回 (首包ms, 音频时长s, 墙钟s, 块数)。不经过 PortAudio。</summary>
        private static async Task<(double firstMs, double durSec, double wallSec, int chunks)> MeasureStreamAsync(
            IStreamTTSService tts, string text, string? voice)
        {
            var options = new TtsSynthesisOptions { Voice = voice, Language = "auto" };
            int chunks = 0;
            long totalSamples = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            double firstMs = double.NaN;

            await foreach (var chunk in tts.GetStreamAudioAsync(text, options).ConfigureAwait(false))
            {
                if (chunk.Samples.Length == 0) continue;
                if (double.IsNaN(firstMs)) firstMs = sw.Elapsed.TotalMilliseconds;
                chunks++;
                totalSamples += chunk.Samples.Length;
            }
            sw.Stop();
            double durSec = (double)totalSamples / SampleRate;
            return (double.IsNaN(firstMs) ? sw.Elapsed.TotalMilliseconds : firstMs, durSec, sw.Elapsed.TotalSeconds, chunks);
        }

        // ---------------------------------------------------------------------
        // 交互模式 (流式播放 / :bench / 切音色)
        // ---------------------------------------------------------------------
        private static async Task RunInteractiveAsync()
        {
            var ttsOptions = BuildTtsOptions(ParseSchedule(Environment.GetEnvironmentVariable("QWEN3_CHUNKS")));
            var playOptions = Options.Create(new PortAudioPlayOptions
            {
                SampleRate = SampleRate,
                ChannelCount = ChannelCount,
                SampleFormat = SampleFormat.Float32,
            });

            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Information);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss.fff "; });
            });

            using var tts = new Qwen3TTSService(Options.Create(ttsOptions), loggerFactory.CreateLogger<Qwen3TTSService>());
            using var player = new PortAudioPlayService(playOptions);

            // 诊断: 订阅播放器事件, 确认"结尾停顿"是否为播放缓冲欠载 (underrun)。
            // playClock 在每次合成开始时重置, 故 [xrun] 的 t=+Xs 即相对该次合成起点的时刻;
            // 若 [xrun] 出现在 [完成] 之后的尾部排空期, 即坐实结尾停顿 = 队列见底插静音。
            var playClock = System.Diagnostics.Stopwatch.StartNew();
            player.OnXrun += (_, e) =>
            {
                if (e.XrunType == AudioXrunType.OutputUnderflow)
                    Console.WriteLine($"  [xrun] 输出欠载 frames={e.FrameCount} 队列={player.CurrentQueueSize} t=+{playClock.Elapsed.TotalSeconds:F2}s");
            };
            player.OnQueueOverflow += (_, e) =>
                Console.WriteLine($"  [overflow] 丢弃={e.DroppedSamples} 样本 队列={e.CurrentQueueSize}/{e.MaxQueueSize}");
            player.OnError += (_, e) =>
                Console.WriteLine($"  [audio-error] {e.Exception.GetType().Name}: {e.Message}");

            var voices = tts.ListVoices();
            Console.WriteLine($"Qwen3-TTS Demo — 采样率 {tts.OutputSampleRate}Hz, {voices.Count} 个内置音色 ({Provider})");
            foreach (var v in voices)
                Console.WriteLine($"  - {v.Name}{(string.IsNullOrEmpty(v.Description) ? "" : $" — {v.Description}")}");
            string? voice = voices.Count > 0 ? voices[0].Name : null;
            string language = "auto";

            await player.StartPlayingAsync();

            Console.WriteLine();
            Console.WriteLine("用法:");
            Console.WriteLine("  直接输入文本回车      -> 流式合成并播放");
            Console.WriteLine("  :voice <名称>         -> 切换音色");
            Console.WriteLine("  :lang <语言>          -> 设置语言 (auto/chinese/english/...)");
            Console.WriteLine("  :bench <文本>         -> 整段合成并报告 RTF");
            Console.WriteLine("  :stop / :q");
            Console.WriteLine();

            CancellationTokenSource? cts = null;
            while (true)
            {
                Console.Write("> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line is ":q" or ":quit") break;
                if (line == ":stop") { cts?.Cancel(); player.ClearQueue(); Console.WriteLine("已停止。"); continue; }
                if (line == ":voices") { foreach (var v in voices) Console.WriteLine($"  {v.Name}"); continue; }
                if (line.StartsWith(":voice ", StringComparison.OrdinalIgnoreCase))
                {
                    string name = line[":voice ".Length..].Trim();
                    var m = voices.FirstOrDefault(v => string.Equals(v.Name, name, StringComparison.OrdinalIgnoreCase));
                    if (m is null) Console.WriteLine($"未找到音色: {name}");
                    else { voice = m.Name; Console.WriteLine($"切换音色 -> {voice}"); }
                    continue;
                }
                if (line.StartsWith(":lang ", StringComparison.OrdinalIgnoreCase))
                {
                    language = line[":lang ".Length..].Trim();
                    Console.WriteLine($"语言 -> {language}");
                    continue;
                }

                cts?.Dispose();
                cts = new CancellationTokenSource();
                try
                {
                    if (line.StartsWith(":bench ", StringComparison.OrdinalIgnoreCase))
                        await BenchAsync(tts, line[":bench ".Length..].Trim(), voice, language, cts.Token);
                    else
                        await SynthesizeAndPlayAsync(tts, player, line, voice, language, playClock, cts.Token);
                }
                catch (OperationCanceledException) { Console.WriteLine("[已取消]"); }
                catch (Exception ex) { Console.WriteLine($"[失败] {ex.GetType().Name}: {ex.Message}"); }
            }

            cts?.Cancel();
            cts?.Dispose();
            await player.StopPlayingAsync();
            Console.WriteLine("再见。");
        }

        private static async Task SynthesizeAndPlayAsync(IStreamTTSService tts, PortAudioPlayService player,
            string text, string? voice, string language, System.Diagnostics.Stopwatch playClock, CancellationToken ct)
        {
            // 播放预缓冲: 先攒够 ~0.5s 再起播, 避免小首块导致 PortAudio 队列欠载 (RTF<1 时缓冲只增不减)。
            const double prerollSeconds = 0.5;
            int prerollThreshold = (int)(prerollSeconds * SampleRate);
            // 配合播放器的 [xrun] 事件, 用于确认"结尾停顿"是否为队列见底欠载。
            var options = new TtsSynthesisOptions { Voice = voice, Language = language };
            int chunks = 0, totalSamples = 0;
            playClock.Restart();
            var start = DateTime.UtcNow;
            DateTime? first = null;

            var prerollBuf = new List<float[]>();
            int prerollSamples = 0;
            bool playing = false;

            // 单点入队 + 日志: 打印入队块时长、入队后队列深度 (块数) 与相对合成起点的时刻。
            void Enqueue(float[] samples)
            {
                player.TryEnqueue(samples, out _);
            }

            await foreach (var chunk in tts.GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.Samples.Length == 0) continue;
                first ??= DateTime.UtcNow;
                chunks++;
                totalSamples += chunk.Samples.Length;

                if (!playing)
                {
                    prerollBuf.Add(chunk.Samples);
                    prerollSamples += chunk.Samples.Length;
                    if (prerollSamples >= prerollThreshold)
                    {
                        foreach (var b in prerollBuf) Enqueue(b);
                        prerollBuf.Clear();
                        playing = true;
                    }
                }
                else
                {
                    Enqueue(chunk.Samples);
                }
            }
            // 不足 preroll 阈值的尾部也要送出
            foreach (var b in prerollBuf) Enqueue(b);

            string latency = first.HasValue ? $"{(first.Value - start).TotalMilliseconds:F0}ms" : "n/a";
            double dur = (double)totalSamples / SampleRate;
            double prodEnd = playClock.Elapsed.TotalSeconds;
            Console.WriteLine($"[完成] 块={chunks} 时长={dur:F2}s 首包={latency} 生成结束 t=+{prodEnd:F2}s");
        }

        private static async Task BenchAsync(ITTSService tts, string text, string? voice, string language, CancellationToken ct)
        {
            var options = new TtsSynthesisOptions { Voice = voice, Language = language };
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = await tts.GetAudioAsync(text, options, ct).ConfigureAwait(false);
            sw.Stop();
            double dur = (double)result.Samples.Length / result.SampleRate;
            double rtf = sw.Elapsed.TotalSeconds / Math.Max(1e-6, dur);
            Console.WriteLine($"[bench] 音频={dur:F2}s 墙钟={sw.Elapsed.TotalSeconds:F2}s RTF={rtf:F3} 音色={result.Voice}");
        }

        // ---------------------------------------------------------------------
        // helpers
        // ---------------------------------------------------------------------
        private static Qwen3TTSOptions BuildTtsOptions(int[]? schedule)
        {
            var o = new Qwen3TTSOptions
            {
                ModelPath = ModelPath,
                ExecutionProvider = Provider,
            };
            if (schedule is { Length: > 0 }) o.StreamChunkSchedule = schedule;
            // QWEN3_IOBINDING=0 forces the talker Dictionary path.
            // QWEN3_CODEC_CPU=1 forces codec_decoder.onnx back to CPU.
            // QWEN3_WARMUP=0 skips warmup. QWEN3_SUBTALKER_SAMPLE=0 tests the greedy CP graph.
            if (Environment.GetEnvironmentVariable("QWEN3_SUBTALKER_SAMPLE") == "0") o.SubtalkerDoSample = false;
            return o;
        }

        private static string? ExtractBenchText(string[] args)
        {
            if (args.Length == 0) return null;
            if (string.Equals(args[0], "--bench", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[0], "bench", StringComparison.OrdinalIgnoreCase))
                return args.Length > 1 ? string.Join(' ', args[1..]) : "这是一个用于测试首包延迟的句子。";
            // 直接把所有参数当作要合成的文本
            return string.Join(' ', args);
        }

        private static int[] ParseSchedule(string? csv)
        {
            if (string.IsNullOrWhiteSpace(csv)) return [];
            var parts = csv.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
            var list = new List<int>(parts.Length);
            foreach (var p in parts)
                if (int.TryParse(p, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0)
                    list.Add(n);
            return list.ToArray();
        }

        private static int? TryGetEnvInt(string name)
            => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : null;
    }
}
