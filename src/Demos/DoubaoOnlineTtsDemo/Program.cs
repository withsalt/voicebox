using Microsoft.Extensions.Logging;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.DoubaoOnline;
using VoiceBox.TTS.DoubaoOnline.SpeakerCatalog;

namespace DoubaoOnlineTtsDemo
{
    /// <summary>
    /// 基于火山引擎「豆包语音合成大模型 2.0 (seed-tts-2.0)」双向流式 WebSocket 接口的实时合成 Demo。
    /// 输入文本 → 流式合成 PCM → PortAudio 边收边播 (低首包延迟)。
    ///
    /// 鉴权 (新版控制台) 通过环境变量提供：
    ///   VOLC_TTS_API_KEY      —— X-Api-Key (新版控制台 API Key)
    ///   VOLC_TTS_RESOURCE_ID  —— 可选，默认 seed-tts-2.0；复刻音色用 seed-icl-2.0
    ///   VOLC_TTS_VOICE        —— 默认音色 (speaker)，如 zh_female_tianmeitaozi_mars_bigtts
    ///
    /// 拉取音色列表 (ListSpeakers，管理类接口，用签名 V4) 可选：
    ///   VOLC_ACCESS_KEY_ID + VOLC_SECRET_ACCESS_KEY  —— 火山引擎账号 AK/SK
    /// </summary>
    internal static class Program
    {
        private const int ChannelCount = 1;

        static async Task Main()
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            var creds = ReadCredentials();
            if (string.IsNullOrWhiteSpace(creds.ApiKey))
            {
                Console.WriteLine("未检测到鉴权配置。请设置环境变量后重试：");
                Console.WriteLine("  新版控制台: VOLC_TTS_API_KEY[, VOLC_TTS_RESOURCE_ID, VOLC_TTS_VOICE]");
                return;
            }

            await RunAsync(creds);
        }

        private const int SampleRate = 24000; // 与合成请求的 sample_rate 保持一致

        private static async Task RunAsync(Credentials creds)
        {
            using var audioServiceProxy = new AudioServiceManager()
                .AddLogger(b =>
                {
                    b.SetMinimumLevel(LogLevel.Information);
                    b.AddSimpleConsole(o =>
                    {
                        o.SingleLine = true;
                        o.TimestampFormat = "HH:mm:ss.fff ";
                    });
                })
                .AddPortAudioPlayService(options =>
                {
                    options.SampleRate = SampleRate;
                    options.ChannelCount = ChannelCount;
                    options.SampleFormat = SampleFormat.Float32;
                })
                .AddDoubaoOnlineTTSService(options =>
                {
                    options.ApiKey = creds.ApiKey;
                    options.ResourceId = creds.ResourceId ?? SeedTtsResourceIds.SeedTts20;
                    options.DefaultVoice = creds.Voice;
                    options.SampleRate = SampleRate;
                    options.ChannelCount = ChannelCount;
                    // 双向流式：不确定服务端何时收尾，给足空闲超时。
                    options.SessionTimeoutSec = 30;
                })
                .Build();

            using var tts = audioServiceProxy.GetStreamTTSServiceInstance();
            using var player = audioServiceProxy.GetAudioPlayInstance();

            // 可选：从 ListSpeakers 拉取音色列表 (需 AK/SK)。
            string? selectedVoice = creds.Voice;
            selectedVoice = await MaybeListSpeakersAsync(creds, selectedVoice);

            Console.WriteLine($"豆包在线 TTS Demo — 采样率 {tts.OutputSampleRate}Hz, 模型 {creds.ResourceId ?? SeedTtsResourceIds.SeedTts20}");
            Console.WriteLine($"当前音色: {selectedVoice ?? "(未设置, 请用 :voice <音色ID> 指定)"}");

            await player.StartPlayingAsync();
            PrintUsage();

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
                if (line.StartsWith(":voice ", StringComparison.OrdinalIgnoreCase))
                {
                    selectedVoice = line[":voice ".Length..].Trim();
                    Console.WriteLine($"切换音色 -> {selectedVoice}");
                    continue;
                }
                if (line.StartsWith(":speakers", StringComparison.OrdinalIgnoreCase))
                {
                    await ListSpeakersAsync(creds);
                    continue;
                }
                if (line.StartsWith(":save ", StringComparison.OrdinalIgnoreCase))
                {
                    string path = line[":save ".Length..].Trim();
                    await SaveAsync(tts, path, selectedVoice);
                    continue;
                }

                if (string.IsNullOrWhiteSpace(selectedVoice))
                {
                    Console.WriteLine("请先用 :voice <音色ID> 指定音色 (或设置 VOLC_TTS_VOICE)。");
                    continue;
                }

                cts?.Dispose();
                cts = new CancellationTokenSource();
                try { await SynthesizeAndPlayAsync(tts, player, line, selectedVoice, cts.Token); }
                catch (OperationCanceledException) { Console.WriteLine("[已取消]"); }
                catch (Exception ex) { Console.WriteLine($"[合成失败] {ex.GetType().Name}: {ex.Message}"); }
            }

            cts?.Cancel();
            cts?.Dispose();
            await player.StopPlayingAsync();
            Console.WriteLine("再见。");
        }

        private static void PrintUsage()
        {
            Console.WriteLine();
            Console.WriteLine("用法:");
            Console.WriteLine("  直接输入文本回车        -> 流式合成并播放");
            Console.WriteLine("  :voice <音色ID>         -> 切换音色 (speaker)");
            Console.WriteLine("  :speakers               -> 拉取并列出可用音色 (需 AK/SK)");
            Console.WriteLine("  :save <文件.wav> <文本>  -> 整段合成并保存为 WAV");
            Console.WriteLine("  :stop                   -> 停止当前播放");
            Console.WriteLine("  :q / :quit              -> 退出");
            Console.WriteLine();
        }

        // 播放端预缓冲: 先攒够 PrerollSeconds 音频再开播, 给播放一个领先量, 避免早期小块欠载卡顿。
        private const double PrerollSeconds = 0.5;

        private static async Task SynthesizeAndPlayAsync(
            IStreamTTSService tts, IAudioPlayService player, string text, string? voice, CancellationToken ct)
        {
            var options = new TtsSynthesisOptions { Voice = voice };
            int chunks = 0, totalFrames = 0;
            var start = DateTime.UtcNow;
            DateTime? first = null;

            int prerollFrames = (int)(PrerollSeconds * SampleRate);
            var preroll = new List<float>(prerollFrames + 4096);
            bool started = false;

            void StartPlayback()
            {
                started = true;
                first ??= DateTime.UtcNow;
                if (preroll.Count > 0) { player.TryEnqueue(preroll.ToArray(), out _); preroll.Clear(); }
            }

            await foreach (var chunk in tts.GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.Samples.Length == 0) continue;
                chunks++;
                totalFrames += chunk.Samples.Length / Math.Max(1, chunk.Channels);
                if (started) player.TryEnqueue(chunk.Samples, out _);
                else
                {
                    preroll.AddRange(chunk.Samples);
                    if (preroll.Count >= prerollFrames) StartPlayback();
                }
            }
            if (!started) StartPlayback();

            string latency = first.HasValue ? $"{(first.Value - start).TotalMilliseconds:F0}ms" : "n/a";
            Console.WriteLine($"[完成] 块={chunks} 时长={(double)totalFrames / SampleRate:F2}s 起播延迟={latency}");
        }

        private static async Task SaveAsync(IStreamTTSService tts, string arg, string? voice)
        {
            int sp = arg.IndexOf(' ');
            if (sp <= 0) { Console.WriteLine("用法: :save <文件.wav> <文本>"); return; }
            string path = arg[..sp].Trim();
            string text = arg[(sp + 1)..].Trim();
            if (string.IsNullOrWhiteSpace(voice)) { Console.WriteLine("请先指定音色。"); return; }

            var result = await tts.GetAudioAsync(text, new TtsSynthesisOptions { Voice = voice });
            WavWriter.Write(path, result.Samples, result.SampleRate, result.Channels);
            Console.WriteLine($"[已保存] {path} — {(double)result.Samples.Length / result.SampleRate:F2}s, 合成耗时 {result.ElapsedSeconds:F2}s");
        }

        private static async Task<string?> MaybeListSpeakersAsync(Credentials creds, string? current)
        {
            if (creds.AccessKeyId is null || creds.SecretAccessKey is null) return current;
            try
            {
                var speakers = await FetchSpeakersAsync(creds);
                if (speakers.Count > 0)
                {
                    Console.WriteLine($"已从 ListSpeakers 拉取 {speakers.Count} 个音色 (输入 :speakers 查看)。");
                    current ??= speakers[0].VoiceType;
                }
            }
            catch (Exception ex) { Console.WriteLine($"[ListSpeakers 跳过] {ex.Message}"); }
            return current;
        }

        private static async Task ListSpeakersAsync(Credentials creds)
        {
            if (creds.AccessKeyId is null || creds.SecretAccessKey is null)
            {
                Console.WriteLine("未配置 VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY，无法拉取音色列表。");
                return;
            }
            try
            {
                var speakers = await FetchSpeakersAsync(creds);
                Console.WriteLine($"共 {speakers.Count} 个音色:");
                foreach (var s in speakers.Take(50))
                    Console.WriteLine($"  {s.VoiceType,-40} {s.Name} ({s.Gender}/{s.Age})");
                if (speakers.Count > 50) Console.WriteLine($"  ... 其余 {speakers.Count - 50} 个略");
            }
            catch (Exception ex) { Console.WriteLine($"[拉取失败] {ex.Message}"); }
        }

        private static async Task<IReadOnlyList<VolcSpeaker>> FetchSpeakersAsync(Credentials creds)
        {
            using var client = new VolcSpeakerCatalogClient(new VolcSpeakerCatalogOptions
            {
                AccessKeyId = creds.AccessKeyId!,
                SecretAccessKey = creds.SecretAccessKey!,
            });
            var result = await client.ListSpeakersAsync(
                resourceIds: new[] { creds.ResourceId ?? SeedTtsResourceIds.SeedTts20 },
                page: 1, limit: 200);
            return result.Speakers;
        }

        private static Credentials ReadCredentials()
        {
            string? Get(string k) => Environment.GetEnvironmentVariable(k) is { Length: > 0 } v ? v : null;
            return new Credentials(
                ApiKey: Get("VOLC_TTS_API_KEY"),
                ResourceId: Get("VOLC_TTS_RESOURCE_ID"),
                Voice: Get("VOLC_TTS_VOICE"),
                AccessKeyId: Get("VOLC_ACCESS_KEY_ID"),
                SecretAccessKey: Get("VOLC_SECRET_ACCESS_KEY"));
        }

        private sealed record Credentials(
            string? ApiKey, string? ResourceId,
            string? Voice, string? AccessKeyId, string? SecretAccessKey);
    }

    /// <summary>写 16-bit PCM WAV (单声道/多声道; frame-major 交错)。</summary>
    internal static class WavWriter
    {
        public static void Write(string path, ReadOnlySpan<float> samples, int sampleRate, int channels)
        {
            const int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;
            int dataSize = samples.Length * sizeof(short);

            var dir = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            using var stream = File.Create(path);
            using var w = new BinaryWriter(stream);
            w.Write("RIFF"u8); w.Write(36 + dataSize); w.Write("WAVE"u8);
            w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)channels);
            w.Write(sampleRate); w.Write(byteRate); w.Write((short)blockAlign); w.Write((short)bitsPerSample);
            w.Write("data"u8); w.Write(dataSize);
            for (int i = 0; i < samples.Length; i++)
                w.Write((short)Math.Round(Math.Clamp(samples[i], -1f, 1f) * 32767f));
        }
    }
}
