using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Runtime.InteropServices;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.Kokoro;

namespace KokoroTtsDemo
{
    internal class Program
    {
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string KokoroModelDir = Path.Combine(BASE_MODEL_DIR, "tts", "kokoro", "kokoro-v1.1");
        private static readonly string ModelPath = Path.Combine(KokoroModelDir, "kokoro-v1.1-zh.onnx");
        private static readonly string VoicesPath = Path.Combine(KokoroModelDir, "voices-v1.1-zh.bin");
        private static readonly string VocabConfigPath = Path.Combine(KokoroModelDir, "config.json");

        private const OnnxExecutionProvider onnxProvider = OnnxExecutionProvider.Cpu;
        private const int SampleRate = 24000;
        private const int ChannelCount = 1;

        static async Task Main(string[] args)
        {
            // Windows 控制台默认 ANSI (CP936) 会把中文读成 "??", espeak-ng 收到后音素为空, 推理返回 0 采样。
            // 显式切 UTF-8 才能正确输入中文。
            if (OperatingSystem.IsWindows())
            {
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (string.IsNullOrWhiteSpace(ModelPath) || string.IsNullOrWhiteSpace(VoicesPath))
            {
                Console.WriteLine("请先配置 ModelPath 和 VoicesPath。");
                Console.WriteLine("  ModelPath:      kokoro-v1.0.onnx 文件路径");
                Console.WriteLine("  VoicesPath:     voices-v1.0.bin 文件路径");
                Console.WriteLine("  VocabConfigPath: config.json 文件路径 (可选)");
                Console.WriteLine();
                Console.WriteLine("下载地址:");
                Console.WriteLine("  https://github.com/thewh1teagle/kokoro-onnx/releases");
                return;
            }

            var kokoroOptions = Options.Create(new KokoroTTSOptions
            {
                ModelPath = ModelPath,
                VoicesPath = VoicesPath,
                VocabConfigPath = string.IsNullOrWhiteSpace(VocabConfigPath) ? null : VocabConfigPath,
                ExecutionProvider = onnxProvider,
                ThreadsNum = Math.Min(Environment.ProcessorCount, 8),
                DefaultSpeed = 1.2f,
                FirstBatchPhonemeBudget = 100,
            });

            var playOptions = Options.Create(new PortAudioPlayOptions
            {
                SampleRate = SampleRate,
                ChannelCount = ChannelCount,
                SampleFormat = SampleFormat.Float32,
            });

            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.SetMinimumLevel(LogLevel.Information);
                builder.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss.fff ";
                });
            });

            using var ttsService = new KokoroTTSService(kokoroOptions, loggerFactory.CreateLogger<KokoroTTSService>());
            using var playService = new PortAudioPlayService(playOptions);

            var voices = ttsService.ListVoices();
            Console.WriteLine($"Kokoro TTS Demo — {voices.Count} voices available");
            Console.WriteLine("可用音色:");
            for (int i = 0; i < voices.Count; i++)
            {
                Console.WriteLine($"  {voices[i].Name}");
            }
            //if (voices.Count > 20)
            //    Console.WriteLine($"  ... ({voices.Count - 20} more)");
            Console.WriteLine();

            string? selectedVoice = "zf_001"; //voices.Count > 0 ? voices[0].Name : null;
            string selectedLanguage = "";
            float selectedSpeed = 1.2f;
            Console.WriteLine($"当前音色: {selectedVoice ?? "(无)"}, 语言: {selectedLanguage}, 语速: {selectedSpeed}");
            Console.WriteLine();

            await playService.StartPlayingAsync();

            Console.WriteLine("用法:");
            Console.WriteLine("  直接输入文本回车           -> 合成并播放");
            Console.WriteLine("  :voice <名称>             -> 切换音色");
            Console.WriteLine("  :lang <en-us|zh|ja|...>   -> 切换语言");
            Console.WriteLine("  :speed <0.5~2.0>          -> 切换语速");
            Console.WriteLine("  :voices                   -> 列出所有音色");
            Console.WriteLine("  :stop                     -> 停止当前播放");
            Console.WriteLine("  :q / :quit                -> 退出");
            Console.WriteLine();

            CancellationTokenSource? currentCts = null;

            while (true)
            {
                Console.Write("> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;

                if (line.Equals(":q", StringComparison.OrdinalIgnoreCase) ||
                    line.Equals(":quit", StringComparison.OrdinalIgnoreCase))
                    break;

                if (line.Equals(":stop", StringComparison.OrdinalIgnoreCase))
                {
                    currentCts?.Cancel();
                    playService.ClearQueue();
                    Console.WriteLine("已停止。");
                    continue;
                }
                if (line.Equals(":voices", StringComparison.OrdinalIgnoreCase))
                {
                    foreach (var v in voices)
                        Console.WriteLine($"  {v.Name}");
                    continue;
                }
                if (line.StartsWith(":voice ", StringComparison.OrdinalIgnoreCase))
                {
                    string name = line[":voice ".Length..].Trim();
                    var matched = voices.FirstOrDefault(v => string.Equals(v.Name, name, StringComparison.OrdinalIgnoreCase));
                    if (matched is null)
                        Console.WriteLine($"未找到音色: {name}");
                    else
                    {
                        selectedVoice = matched.Name;
                        Console.WriteLine($"切换音色 -> {selectedVoice}");
                    }
                    continue;
                }
                if (line.StartsWith(":lang ", StringComparison.OrdinalIgnoreCase))
                {
                    selectedLanguage = line[":lang ".Length..].Trim();
                    Console.WriteLine($"切换语言 -> {selectedLanguage}");
                    continue;
                }
                if (line.StartsWith(":speed ", StringComparison.OrdinalIgnoreCase))
                {
                    if (float.TryParse(line[":speed ".Length..].Trim(), out float spd) && spd >= 0.5f && spd <= 2.0f)
                    {
                        selectedSpeed = spd;
                        Console.WriteLine($"切换语速 -> {selectedSpeed}");
                    }
                    else
                        Console.WriteLine("语速范围: 0.5 ~ 2.0");
                    continue;
                }

                currentCts?.Dispose();
                currentCts = new CancellationTokenSource();

                try
                {
                    await SynthesizeAndPlayAsync(ttsService, playService, line, selectedVoice, selectedLanguage, selectedSpeed, currentCts.Token);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[已取消]");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[合成失败] {ex.Message}");
                }
            }

            currentCts?.Cancel();
            currentCts?.Dispose();
            await playService.StopPlayingAsync();
            Console.WriteLine("再见。");
        }

        private static async Task SynthesizeAndPlayAsync(
            IStreamTTSService tts,
            PortAudioPlayService player,
            string text,
            string? voice,
            string language,
            float speed,
            CancellationToken ct)
        {
            var options = new TtsSynthesisOptions
            {
                Voice = voice,
                Language = language,
                Speed = speed,
            };

            int chunkCount = 0;
            int totalSamples = 0;
            DateTime startUtc = DateTime.UtcNow;
            DateTime? firstAudioUtc = null;

            await foreach (var chunk in tts.GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.Samples.Length == 0) continue;
                if (firstAudioUtc is null) firstAudioUtc = DateTime.UtcNow;

                player.TryEnqueue(chunk.Samples, out _);
                chunkCount++;
                totalSamples += chunk.Samples.Length;

                if (chunk.IsFinal) break;
            }

            double duration = (double)totalSamples / SampleRate;
            string latency = firstAudioUtc.HasValue
                ? $"{(firstAudioUtc.Value - startUtc).TotalMilliseconds:F0}ms"
                : "n/a";
            Console.WriteLine($"[完成] 块={chunkCount} 时长={duration:F2}s 首包延迟={latency}");
        }
    }
}
