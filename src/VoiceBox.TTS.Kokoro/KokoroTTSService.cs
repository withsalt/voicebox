using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Misaki.NET.Lang.Chinese;
using Misaki.NET.Lang.English;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.Kokoro.Internal;
using WeTextProcessing.NET;

namespace VoiceBox.TTS.Kokoro
{
    public sealed class KokoroTTSService : IStreamTTSService
    {
        private readonly KokoroTTSOptions _options;
        private readonly ILogger<KokoroTTSService> _logger;
        private readonly KokoroInferenceSession _session;
        private readonly NpzVoiceStore _voiceStore;
        private readonly KokoroTokenizer _tokenizer;
        private readonly EnglishG2P _engPhonemizer;
        private readonly ChineseG2P _chinesePhonemizer;
        private readonly WetextProcessor? _chineseTextNormalizer;
        private readonly WetextProcessor? _englishTextNormalizer;
        private readonly object _textNormalizationLock = new();
        private readonly SemaphoreSlim _gate;
        private readonly string _defaultVoice;
        // 服务级关闭信号: Dispose 时取消, 让在途流式合成经链接 token 及时停止 (与 Qwen3TTSService 一致)。
        private readonly CancellationTokenSource _shutdownCts = new();
        private int _disposed;

        public KokoroTTSService(IOptions<KokoroTTSOptions> options, ILogger<KokoroTTSService>? logger = null)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<KokoroTTSService>.Instance;

            ValidateOptions(_options);

            _session = new KokoroInferenceSession(_options.ModelPath, _options);
            _voiceStore = NpzVoiceStore.Load(_options.VoicesPath);

            _tokenizer = !string.IsNullOrEmpty(_options.VocabConfigPath)
                ? new KokoroTokenizer(_options.VocabConfigPath)
                : new KokoroTokenizer();

            _engPhonemizer = new EnglishG2P();
            _chinesePhonemizer = new ChineseG2P(version: "1.1", enCallable: en => _engPhonemizer.Process(en).Phonemes);
            _chineseTextNormalizer = CreateTextNormalizer(WetextLanguage.Chinese);
            _englishTextNormalizer = CreateTextNormalizer(WetextLanguage.English);
            _gate = new SemaphoreSlim(_options.MaxConcurrentSyntheses, _options.MaxConcurrentSyntheses);

            var voices = _voiceStore.GetVoiceNames();
            if (voices.Count == 0)
                throw new InvalidDataException("Voice store is empty.");

            _defaultVoice = !string.IsNullOrWhiteSpace(_options.DefaultVoice) && _voiceStore.HasVoice(_options.DefaultVoice)
                ? _options.DefaultVoice
                : voices[0];

            try
            {
                // 中文 / 英文 G2P 都做一次 dummy，触发 Jieba 词典 / PinyinLookup 的延迟加载，
                // 避免首次合成时把 ~150ms 的初始化代价压在第一句上。
                _ = _chinesePhonemizer.Process("你好").Phonemes;
                _ = _engPhonemizer.Process("hello").Phonemes;

                var (style, shape) = _voiceStore.GetAnyVoiceStyle();
                _session.Warmup(style, shape);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Kokoro warmup failed; first request may be slower.");
            }

            _logger.LogDebug("KokoroTTS initialized: voices={VoiceCount}, default={DefaultVoice}", voices.Count, _defaultVoice);
        }

        public int OutputSampleRate => KokoroInferenceSession.SampleRate;

        public int OutputChannels => 1;

        public IReadOnlyList<TtsVoice> ListVoices()
        {
            EnsureNotDisposed();
            var names = _voiceStore.GetVoiceNames();
            var result = new List<TtsVoice>(names.Count);
            foreach (string name in names)
            {
                result.Add(new TtsVoice { Name = name });
            }
            return result;
        }

        public TtsVoice? GetDefaultVoice()
        {
            EnsureNotDisposed();
            return new TtsVoice { Name = _defaultVoice };
        }

        public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
        {
            EnsureNotDisposed();
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("text cannot be empty.", nameof(text));

            string voiceName = ResolveVoiceName(options?.Voice);
            long startTicks = Environment.TickCount64;
            var allChunks = new List<float[]>();

            await foreach (var chunk in GetStreamAudioAsync(text, options, ct).ConfigureAwait(false))
            {
                if (chunk.Samples.Length > 0)
                    allChunks.Add(chunk.Samples);
            }

            int totalLength = 0;
            foreach (var c in allChunks) totalLength += c.Length;

            float[] merged = new float[totalLength];
            int offset = 0;
            foreach (var c in allChunks)
            {
                Array.Copy(c, 0, merged, offset, c.Length);
                offset += c.Length;
            }

            if (merged.Length > 0)
                merged = AudioTrimmer.TrimSilence(merged);

            return new TtsResult
            {
                Samples = merged,
                SampleRate = KokoroInferenceSession.SampleRate,
                Channels = 1,
                Voice = voiceName,
                ElapsedSeconds = (Environment.TickCount64 - startTicks) / 1000.0,
            };
        }

        public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
            string text,
            TtsSynthesisOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            EnsureNotDisposed();
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("text cannot be empty.", nameof(text));

            // 调用方 ct 与服务级 shutdown 链接: Dispose 取消 _shutdownCts 时在途合成及时停止。
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
            var token = linked.Token;

            var channel = Channel.CreateUnbounded<TtsAudioChunk>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
                AllowSynchronousContinuations = false,
            });

            var generationTask = Task.Run(() => RunSynthesis(text, options, channel.Writer, token), token);

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(token).ConfigureAwait(false))
                {
                    yield return chunk;
                }
            }
            finally
            {
                // 消费结束 (正常完成 / 异常 / 提前 break / barge-in): 取消并 join 生产者, 再释放 linked。
                // 不 join 会让生产者成为孤儿任务 (终态异常无人观测); 释放 linked 解除其在 _shutdownCts
                // 上的注册, 否则长跑会逐请求累积注册。
                linked.Cancel();
                try { await generationTask.ConfigureAwait(false); }
                catch { /* 生产者错误已经过 channel 传给消费方; 这里只保证 join, 不掩盖原异常 */ }
                linked.Dispose();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            // 先发关闭信号, 让在途流式合成经链接 token 尽快退出并释放 gate。
            try { _shutdownCts.Cancel(); } catch { /* ignore */ }
            try
            {
                for (int i = 0; i < _options.MaxConcurrentSyntheses; i++)
                    _gate.Wait(TimeSpan.FromSeconds(30));
            }
            catch { /* ignore, continue cleanup */ }
            _gate.Dispose();
            _session.Dispose();
            _engPhonemizer?.Dispose();
            _chineseTextNormalizer?.Dispose();
            _englishTextNormalizer?.Dispose();
            _shutdownCts.Dispose();
        }

        private void RunSynthesis(string text, TtsSynthesisOptions? options, ChannelWriter<TtsAudioChunk> writer, CancellationToken ct)
        {
            Exception? failure = null;
            try
            {
                _gate.Wait(ct);
                try
                {
                    SynthesizeStreaming(text, options, writer, ct);
                }
                finally
                {
                    _gate.Release();
                }
            }
            catch (Exception ex)
            {
                failure = ex;
                _logger.LogError(ex, "Kokoro TTS synthesis failed.");
            }
            finally
            {
                writer.TryComplete(failure);
            }
        }

        /// <summary>
        /// 流水线合成: 按句子切原文，G2P 与 ONNX 推理在两个线程上并行；
        /// 第一句 G2P 完成即可立即开始推理，显著降低首块延迟。
        /// </summary>
        private void SynthesizeStreaming(string text, TtsSynthesisOptions? options, ChannelWriter<TtsAudioChunk> writer, CancellationToken ct)
        {
            string voiceName = ResolveVoiceName(options?.Voice);
            float speed = ResolveSpeed(options);
            text = NormalizeText(text, voiceName);
            var sentences = SplitSentences(text);
            if (sentences.Count == 0) return;

            // 生产者: 逐句 G2P → 拆批 → 推入 batch 队列
            var batchChannel = Channel.CreateBounded<PhonemeBatch>(new BoundedChannelOptions(8)
            {
                SingleReader = true,
                SingleWriter = true,
                FullMode = BoundedChannelFullMode.Wait,
            });

            var producer = Task.Run(() =>
            {
                // 生产者把自身异常通过 batchChannel.Writer.TryComplete(error) 传给消费者:
                // 本后端 producer 写的是独立的 batchChannel (而非直接写 writer), 若不把 error 交给
                // batchChannel, 消费者 WaitToReadAsync 只会在排空后返回 false (不抛), G2P/拆批异常
                // (如 Phonemize 在第 2 句抛错) 会被吞掉, 末尾音频丢失却仍写出 IsFinal=true 的“成功”尾块。
                // 完成时带上 error: 消费者先排空已入队的 batch, 再由 WaitToReadAsync 抛出该异常,
                // 冒泡到 RunSynthesis -> writer.TryComplete(failure), 与其他流式后端单一错误路径一致。
                Exception? producerError = null;
                try
                {
                    bool isFirstBatch = true;
                    for (int sIdx = 0; sIdx < sentences.Count; sIdx++)
                    {
                        ct.ThrowIfCancellationRequested();
                        string raw = Phonemize(sentences[sIdx]);
                        string phonemes = _tokenizer.FilterPhonemes(raw);
                        if (string.IsNullOrEmpty(phonemes)) continue;

                        // 第一批用短预算降低首块延迟，后续句不再限制。
                        int budget = isFirstBatch ? _options.FirstBatchPhonemeBudget : 0;
                        var batches = PhonemeSplitter.Split(phonemes, budget);
                        for (int bIdx = 0; bIdx < batches.Count; bIdx++)
                        {
                            bool isLastBatch = sIdx == sentences.Count - 1 && bIdx == batches.Count - 1;
                            var batch = new PhonemeBatch(batches[bIdx], isLastBatch);
                            batchChannel.Writer.WriteAsync(batch, ct).AsTask().GetAwaiter().GetResult();
                            isFirstBatch = false;
                        }
                    }
                }
                catch (Exception ex)
                {
                    // 捕获后交给 channel 完成信号传递; producer 任务自身不再 fault (避免孤儿未观测异常),
                    // ChannelClosedException (消费方先 TryComplete 解阻塞 WriteAsync 的情形) 无需上报。
                    producerError = ex;
                }
                finally
                {
                    batchChannel.Writer.TryComplete(producerError);
                }
            }, ct);

            // 消费者: 逐批推理写出。在主任务线程执行。
            long emittedSamplesTotal = 0;

            try
            {
                while (batchChannel.Reader.WaitToReadAsync(ct).AsTask().GetAwaiter().GetResult())
                {
                    ct.ThrowIfCancellationRequested();

                    while (batchChannel.Reader.TryRead(out var batch))
                    {
                        float[] audio = SynthesizeBatch(batch.Phonemes, voiceName, speed, batch.IsLast);
                        if (audio.Length == 0)
                            continue;

                        emittedSamplesTotal += audio.Length;
                        writer.TryWrite(new TtsAudioChunk
                        {
                            Samples = audio,
                            SampleRate = KokoroInferenceSession.SampleRate,
                            Channels = 1,
                            EmittedSeconds = (double)emittedSamplesTotal / KokoroInferenceSession.SampleRate,
                            IsFinal = false,
                        });
                    }
                }
            }
            finally
            {
                // 异常/取消路径也要 join 生产者, 否则 producer 成为孤儿任务, 与其他流式后端对齐。
                // 先 TryComplete 写端: 当消费方抛出非取消异常时, producer 可能正阻塞在 bounded channel 的
                // WriteAsync 等读方消费, 直接 GetResult 会死锁; 完成写端让该 WriteAsync 抛 ChannelClosedException
                // 退出, 再 join 即可。producer 已把自身异常经 batchChannel 完成信号传给消费方 (见上), 任务本身
                // 正常结束, 故此处 GetResult 通常不抛; catch 仅防御 (如任务启动前即被 ct 取消)。
                batchChannel.Writer.TryComplete();
                try { producer.GetAwaiter().GetResult(); }
                catch (OperationCanceledException) { /* 取消是预期退出 */ }
                catch { /* producer 异常已经 channel 传给消费方并冒泡; 不在 join 处重复抛出 */ }
            }

            writer.TryWrite(new TtsAudioChunk
            {
                Samples = Array.Empty<float>(),
                SampleRate = KokoroInferenceSession.SampleRate,
                Channels = 1,
                EmittedSeconds = (double)emittedSamplesTotal / KokoroInferenceSession.SampleRate,
                IsFinal = true,
            });
        }

        private readonly record struct PhonemeBatch(string Phonemes, bool IsLast);

        private float[] SynthesizeBatch(string phonemeBatch, string voiceName, float speed, bool isLastBatch)
        {
            int[] tokens = _tokenizer.Tokenize(phonemeBatch);
            var (voiceStyle, styleShape) = _voiceStore.GetVoiceStyle(voiceName, tokens.Length);
            float[] audio = _session.Synthesize(tokens, voiceStyle, styleShape, speed);
            if (isLastBatch)
            {
                audio = AudioTrimmer.TrimSilence(audio);
            }
            return audio;
        }

        private string Phonemize(string text)
        {
            return ContainsChineseChar(text)
                ? _chinesePhonemizer.Process(text).Phonemes
                : _engPhonemizer.Process(text).Phonemes;
        }

        private WetextProcessor? CreateTextNormalizer(WetextLanguage language)
        {
            if (!_options.EnableTextNormalization) return null;

            try
            {
                return new WetextProcessor(language, WetextMode.TextNormalization);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "Kokoro {Language} text normalizer initialization failed; matching text will be passed through.",
                    language);
                return null;
            }
        }

        private string NormalizeText(string text, string voiceName)
        {
            if (!_options.EnableTextNormalization) return text;

            bool useChinese = ContainsChineseChar(text)
                || (!ContainsLatinLetter(text) && IsChineseVoice(voiceName));
            WetextProcessor? normalizer = useChinese
                ? _chineseTextNormalizer
                : _englishTextNormalizer;
            if (normalizer is null) return text;

            try
            {
                lock (_textNormalizationLock)
                {
                    return normalizer.Normalize(text);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Kokoro text normalization failed; using original text.");
                return text;
            }
        }

        private static bool ContainsLatinLetter(string text)
        {
            foreach (char c in text)
            {
                if (c is >= 'A' and <= 'Z' or >= 'a' and <= 'z') return true;
            }
            return false;
        }

        private static bool IsChineseVoice(string voiceName)
            => voiceName.StartsWith("z", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// 按中英文标点把原文切成句子，用于 G2P 流水线的最小工作单元。
        /// 单句过长时直接整体返回，由 PhonemeSplitter 在 phoneme 层再切。
        /// </summary>
        private static List<string> SplitSentences(string text)
        {
            var result = new List<string>();
            int start = 0;
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                bool isBreak = c is '.' or '!' or '?' or ';' or '\n'
                    or '。' or '！' or '？' or '；'
                    or ',' or '，' or '、';
                if (!isBreak) continue;

                string seg = text[start..(i + 1)].Trim();
                if (seg.Length > 0) result.Add(seg);
                start = i + 1;
            }
            if (start < text.Length)
            {
                string tail = text[start..].Trim();
                if (tail.Length > 0) result.Add(tail);
            }
            if (result.Count == 0) result.Add(text);
            return result;
        }

        private string ResolveVoiceName(string? voice)
        {
            if (string.IsNullOrWhiteSpace(voice)) return _defaultVoice;
            if (!_voiceStore.HasVoice(voice))
                throw new ArgumentException($"Voice '{voice}' not found.", nameof(voice));
            return voice;
        }

        private float ResolveSpeed(TtsSynthesisOptions? options)
        {
            float speed = options?.Speed ?? _options.DefaultSpeed;
            if (!float.IsFinite(speed) || speed < 0.5f || speed > 2.0f)
                throw new ArgumentOutOfRangeException(nameof(options), "Speed must be between 0.5 and 2.0.");
            return speed;
        }

        private static bool ContainsChineseChar(string text)
        {
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                if (c >= '一' && c <= '鿿') return true;
                if (c >= '㐀' && c <= '䶿') return true;
            }
            return false;
        }

        private static void ValidateOptions(KokoroTTSOptions options)
        {
            if (string.IsNullOrWhiteSpace(options.ModelPath))
                throw new ArgumentException("ModelPath cannot be empty.");
            if (!File.Exists(options.ModelPath))
                throw new FileNotFoundException($"Model file not found: {options.ModelPath}");
            if (string.IsNullOrWhiteSpace(options.VoicesPath))
                throw new ArgumentException("VoicesPath cannot be empty.");
            if (!File.Exists(options.VoicesPath))
                throw new FileNotFoundException($"Voices file not found: {options.VoicesPath}");
            if (options.MaxConcurrentSyntheses < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentSyntheses must be >= 1.");
            if (!float.IsFinite(options.DefaultSpeed) || options.DefaultSpeed < 0.5f || options.DefaultSpeed > 2.0f)
                throw new ArgumentOutOfRangeException(nameof(options), "DefaultSpeed must be between 0.5 and 2.0.");
        }

        private void EnsureNotDisposed()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        }
    }
}
