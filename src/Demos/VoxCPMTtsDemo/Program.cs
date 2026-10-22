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
using VoiceBox.TTS.VoxCPM;

namespace VoxCPMTtsDemo
{
    internal static class Program
    {
        // VoxCPM 包目录。默认取仓库内导出产物 (fp32 与 PyTorch 对拍 SNR≥80dB; int8 体积最小)。
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "tts", "voxcpm", "voxcpm-0.5b-onnx-int8");
        private static readonly OnnxExecutionProvider Provider = OnnxExecutionProvider.Cpu;

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
                Console.WriteLine("请先运行: python scripts/voxcpm-tts-onnx-export/export.py --dtype fp32");
                return;
            }

            await RunInteractiveAsync();
        }

        private static VoxCPMTTSOptions BuildTtsOptions()
        {
            var o = new VoxCPMTTSOptions
            {
                ModelPath = ModelPath,
                ExecutionProvider = Provider,
                DefaultSeed = TryGetEnvInt("VOXCPM_SEED") ?? 3,
            };
            // 克隆音色: ① VOXCPM_PROMPT_WAV/VOXCPM_PROMPT_TEXT 注册 "custom";
            // ② golden 目录的示例 prompt (若存在) 注册 "example"。
            o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "sichuan_female", PromptWavPath = "./voices/sichuan_female.wav", PromptText = "你晓不晓得，今天天气巴适得很，我准备出去逛一哈，买点菜，晚上回来整火锅。" });
            o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "sichuan_male", PromptWavPath = "./voices/sichuan_male.wav", PromptText = "你晓不晓得，今天天气巴适得很，我准备出去逛一哈，买点菜，晚上回来整火锅。" });
            o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "xiaoxiao_female", PromptWavPath = "./voices/xiaoxiao_female.wav", PromptText = "你好，很高兴认识你。今天天气不错，我准备去公园散步，看看花草树木，也听听孩子们的笑声。" });
            o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "yuanfang_male", PromptWavPath = "./voices/yuanfang_male.wav", PromptText = "你好，很高兴认识你。今天天气不错，我准备去公园散步，看看花草树木，也听听孩子们的笑声。" });
            o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "muyi", PromptWavPath = "./voices/muyi.wav", PromptText = "你好，很高兴认识你。今天天气不错，我准备去公园散步，看看花草树木，也听听孩子们的笑声。" });

            return o;
        }

        // ---------------------------------------------------------------------
        // 交互模式 (流式播放)
        // ---------------------------------------------------------------------
        private static async Task RunInteractiveAsync()
        {
            using var loggerFactory = LoggerFactory.Create(b =>
            {
                b.SetMinimumLevel(LogLevel.Information);
                b.AddSimpleConsole(o => { o.SingleLine = true; o.TimestampFormat = "HH:mm:ss.fff "; });
            });

            using var tts = new VoxCPMTTSService(Options.Create(BuildTtsOptions()),
                loggerFactory.CreateLogger<VoxCPMTTSService>());
            int sampleRate = tts.OutputSampleRate;

            var playOptions = Options.Create(new PortAudioPlayOptions
            {
                SampleRate = (uint)sampleRate,
                ChannelCount = ChannelCount,
                SampleFormat = SampleFormat.Float32,
            });
            using var player = new PortAudioPlayService(playOptions);

            var playClock = System.Diagnostics.Stopwatch.StartNew();
            player.OnXrun += (_, e) =>
            {
                if (e.XrunType == AudioXrunType.OutputUnderflow)
                    Console.WriteLine($"  [xrun] 输出欠载 frames={e.FrameCount} 队列={player.CurrentQueueSize} t=+{playClock.Elapsed.TotalSeconds:F2}s");
            };
            player.OnError += (_, e) =>
                Console.WriteLine($"  [audio-error] {e.Exception.GetType().Name}: {e.Message}");

            var voices = tts.ListVoices();
            Console.WriteLine($"VoxCPM-TTS Demo — {sampleRate}Hz ({Provider})");
            foreach (var v in voices)
                Console.WriteLine($"  - {v.Name}{(string.IsNullOrEmpty(v.Description) ? "" : $" — {v.Description}")}");
            string? voice = "xiaoxiao_female";
            int? seed = TryGetEnvInt("VOXCPM_SEED");

            await player.StartPlayingAsync();

            Console.WriteLine();
            Console.WriteLine("用法:");
            Console.WriteLine("  直接输入文本回车      -> 流式合成并播放 (每 80ms 一块)");
            Console.WriteLine("  :voice <名称>         -> 切换音色 (default = 无 prompt)");
            Console.WriteLine("  :seed <整数|off>      -> 固定/取消随机种子");
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
                if (line == ":voices") { foreach (var v in tts.ListVoices()) Console.WriteLine($"  {v.Name}"); continue; }
                if (line.StartsWith(":voice ", StringComparison.OrdinalIgnoreCase))
                {
                    string name = line[":voice ".Length..].Trim();
                    var m = tts.ListVoices().FirstOrDefault(v => string.Equals(v.Name, name, StringComparison.OrdinalIgnoreCase));
                    if (m is null) Console.WriteLine($"未找到音色: {name}");
                    else { voice = m.Name; Console.WriteLine($"切换音色 -> {voice}"); }
                    continue;
                }
                if (line.StartsWith(":seed ", StringComparison.OrdinalIgnoreCase))
                {
                    string s = line[":seed ".Length..].Trim();
                    seed = int.TryParse(s, out int n) ? n : null;
                    Console.WriteLine(seed.HasValue ? $"种子 -> {seed}" : "种子 -> 随机");
                    continue;
                }

                cts?.Dispose();
                cts = new CancellationTokenSource();
                try
                {
                    if (line.StartsWith(":bench ", StringComparison.OrdinalIgnoreCase))
                        await BenchAsync(tts, line[":bench ".Length..].Trim(), voice, seed, cts.Token);
                    else
                        await SynthesizeAndPlayAsync(tts, player, line, voice, seed, sampleRate, playClock, cts.Token);
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
            string text, string? voice, int? seed, int sampleRate, System.Diagnostics.Stopwatch playClock, CancellationToken ct)
        {
            // 播放预缓冲: 先攒够 ~0.5s 再起播 (RTF>1 的机器上防欠载; RTF<1 时只增不减)。
            const double prerollSeconds = 0.5;
            int prerollThreshold = (int)(prerollSeconds * sampleRate);
            var options = new TtsSynthesisOptions { Voice = voice, Seed = seed };
            int chunks = 0, totalSamples = 0;
            playClock.Restart();
            var start = DateTime.UtcNow;
            DateTime? first = null;

            var prerollBuf = new List<float[]>();
            int prerollSamples = 0;
            bool playing = false;

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
                        foreach (var b in prerollBuf) player.TryEnqueue(b, out _);
                        prerollBuf.Clear();
                        playing = true;
                    }
                }
                else
                {
                    player.TryEnqueue(chunk.Samples, out _);
                }
            }
            foreach (var b in prerollBuf) player.TryEnqueue(b, out _);

            string latency = first.HasValue ? $"{(first.Value - start).TotalMilliseconds:F0}ms" : "n/a";
            double dur = (double)totalSamples / sampleRate;
            Console.WriteLine($"[完成] 块={chunks} 时长={dur:F2}s 首包={latency} 生成结束 t=+{playClock.Elapsed.TotalSeconds:F2}s");
        }

        private static async Task BenchAsync(ITTSService tts, string text, string? voice, int? seed, CancellationToken ct)
        {
            var options = new TtsSynthesisOptions { Voice = voice, Seed = seed };
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
        private static string? ExtractBenchText(string[] args)
        {
            if (args.Length == 0) return null;
            if (string.Equals(args[0], "--bench", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[0], "bench", StringComparison.OrdinalIgnoreCase))
                return args.Length > 1 ? string.Join(' ', args[1..]) : "这是一个用于测试首包延迟的句子。";
            return string.Join(' ', args);
        }

        private static int? TryGetEnvInt(string name)
            => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : null;

        private static string FindRepoRoot()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                if (File.Exists(Path.Combine(dir.FullName, "src", "VoiceBox.slnx")))
                    return dir.FullName;
                dir = dir.Parent!;
            }
            return AppContext.BaseDirectory;
        }
    }
}
