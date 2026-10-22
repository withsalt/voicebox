using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.MossNano;
using WeTextProcessing.NET;

namespace MossNanoTtsDemo
{
    internal class Program
    {
        // MOSS-TTS-Nano-100M-ONNX 模型根目录。同级目录需有 MOSS-Audio-Tokenizer-Nano-ONNX。
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "tts", "open-moss", "MOSS-TTS-Nano-100M-ONNX");

        // codec 输出固定为 48 kHz 立体声 float32。
        private const int SampleRate = 48000;
        private const int ChannelCount = 2;

        private static readonly OnnxExecutionProvider OnnxProvider = OnnxExecutionProvider.Cpu;

        static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (!Directory.Exists(ModelPath))
            {
                Console.WriteLine($"模型目录不存在: {ModelPath}");
                Console.WriteLine("请从 https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX 下载 MOSS-TTS-Nano-100M-ONNX,");
                Console.WriteLine("以及 https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX 下载 MOSS-Audio-Tokenizer-Nano-ONNX,");
                Console.WriteLine("放置在同一上级目录下。");
                return;
            }

            var mossOptions = new MossNanoTTSOptions
            {
                ModelPath = ModelPath,
                ExecutionProvider = OnnxProvider,
                StreamingChunkFrames = 2,
                ThreadsNum = 4,
            };
            var ttsOptions = Options.Create(mossOptions);

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

            using var wetextProcessor = new WetextProcessor();

            using var ttsService = new MossNanoTTSService(ttsOptions, loggerFactory.CreateLogger<MossNanoTTSService>());
            using var playService = new PortAudioPlayService(playOptions);

            var voices = ttsService.ListVoices();
            Console.WriteLine($"MOSS-TTS-Nano Demo — {voices.Count} voices available");
            Console.WriteLine("可用音色 (前 20 个):");
            for (int i = 0; i < Math.Min(20, voices.Count); i++)
            {
                var v = voices[i];
                string desc = string.IsNullOrEmpty(v.Description) ? string.Empty : $" — {v.Description}";
                Console.WriteLine($"  {i + 1}. {v.Name}{desc}");
            }
            Console.WriteLine();

            string? selectedVoice = "Xiaoyu"; //voices.Count > 0 ? voices[0].Name : null;
            Console.WriteLine($"默认音色: {selectedVoice ?? "(无)"}");
            Console.WriteLine();

            await playService.StartPlayingAsync();

            Console.WriteLine("用法:");
            Console.WriteLine("  直接输入文本回车           -> 流式合成并播放");
            Console.WriteLine("  :voice <名称>             -> 切换内置音色");
            Console.WriteLine("  :voices                   -> 列出全部音色");
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
                    {
                        string desc = string.IsNullOrEmpty(v.Description) ? string.Empty : $" — {v.Description}";
                        Console.WriteLine($"  {v.Name}{desc}");
                    }
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

                currentCts?.Dispose();
                currentCts = new CancellationTokenSource();

                try
                {
                    //var normalized = wetextProcessor.Normalize(line);
                    await SynthesizeAndPlayAsync(ttsService, playService, line, selectedVoice, currentCts.Token);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[已取消]");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[合成失败] {ex.GetType().Name}: {ex.Message}");
                }
            }

            currentCts?.Cancel();
            currentCts?.Dispose();
            await playService.StopPlayingAsync();
            wetextProcessor?.Dispose();
            Console.WriteLine("再见。");
        }

        private static async Task SynthesizeAndPlayAsync(
            IStreamTTSService tts,
            PortAudioPlayService player,
            string text,
            string? voice,
            CancellationToken ct)
        {
            var options = new TtsSynthesisOptions { Voice = voice };

            int chunkCount = 0;
            int totalFrames = 0;
            var startUtc = DateTime.UtcNow;
            DateTime? firstAudioUtc = null;

            await foreach (var chunk in tts.GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.IsFinal) break;
                if (chunk.Samples.Length == 0) continue;
                if (firstAudioUtc is null) firstAudioUtc = DateTime.UtcNow;

                player.TryEnqueue(chunk.Samples, out _);
                chunkCount++;
                totalFrames += chunk.Samples.Length / Math.Max(1, chunk.Channels);
            }

            double duration = (double)totalFrames / SampleRate;
            string latency = firstAudioUtc.HasValue
                ? $"{(firstAudioUtc.Value - startUtc).TotalMilliseconds:F0}ms"
                : "n/a";
            Console.WriteLine($"[完成] 块={chunkCount} 时长={duration:F2}s 首包延迟={latency}");
        }
    }
}
