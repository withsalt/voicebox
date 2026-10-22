using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;
using VoiceBox.Core.Onnx;
using WeTextProcessing.NET;

namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// MOSS-TTS-Nano 高层流水线。组合 ModelBundle / InferenceEngine / AudioFrameGenerator / 文本切分 / 流式解码,
    /// 为 <see cref="MossNanoTTSService"/> 提供 voice-clone 整段合成与流式合成入口。
    /// 与 Python `OnnxTtsRuntime.synthesize` 行为 1:1 对齐。
    /// </summary>
    internal sealed class MossNanoPipeline : IDisposable
    {
        private readonly ModelBundle _bundle;
        private readonly InferenceEngine _engine;
        private readonly AudioFrameGenerator _generator;
        private readonly CodecStreamingDecodeSession _streamingSession;
        private readonly Tokenizer _tokenizer;
        private readonly ILogger _logger;
        private readonly WetextProcessor? _chineseTextNormalizer;
        private readonly WetextProcessor? _englishTextNormalizer;
        private readonly object _textNormalizationLock = new();
        private readonly SessionOptions? _ownedTtsSessionOptions;
        private readonly bool _ownsTtsSessionOptions;

        public int SampleRate => _bundle.CodecMeta.CodecConfig.SampleRate;
        public int Channels => _bundle.CodecMeta.CodecConfig.Channels;
        public IReadOnlyList<ManifestBuiltinVoice> BuiltinVoices => _bundle.Manifest.BuiltinVoices;
        public ManifestGenerationDefaults DefaultsFromManifest => _bundle.Manifest.GenerationDefaults;

        public MossNanoPipeline(string modelPath, MossNanoTTSOptions options, ILogger? logger = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
            ArgumentNullException.ThrowIfNull(options);
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;

            string ttsDir = Path.GetFullPath(modelPath);
            if (!Directory.Exists(ttsDir))
                throw new DirectoryNotFoundException($"模型目录不存在: {ttsDir}");

            string manifestPath = Path.Combine(ttsDir, "browser_poc_manifest.json");
            if (!File.Exists(manifestPath))
                throw new FileNotFoundException($"browser_poc_manifest.json 不存在: {manifestPath}", manifestPath);

            // ---- 解析 manifest ----
            var manifest = LoadJson<ModelManifest>(manifestPath);

            string ttsMetaPath = ResolveRelativeFile(ttsDir, manifest.ModelFiles.TtsMeta);
            string codecMetaPath = ResolveRelativeFile(ttsDir, manifest.ModelFiles.CodecMeta);

            var ttsMeta = LoadJson<TtsMeta>(ttsMetaPath);
            var codecMeta = LoadJson<CodecMeta>(codecMetaPath);

            string codecDir = Path.GetDirectoryName(codecMetaPath)
                ?? throw new InvalidDataException("codec_meta 路径无父目录。");

            // ---- 创建 SessionOptions ----
            var ttsSessionOptions = OnnxSessionFactory.Create(options);
            _ownedTtsSessionOptions = ttsSessionOptions;
            _ownsTtsSessionOptions = options.SessionOptionsFactory is null;

            // ---- 加载 ONNX session ----
            _bundle = new ModelBundle(manifest, ttsMeta, codecMeta, ttsDir, codecDir,
                ttsSessionOptions, ttsSessionOptions);

            // ---- 加载 SentencePiece tokenizer ----
            // addBeginningOfSentence / addEndOfSentence 都关掉, 与 Python sentencepiece 库的默认行为一致;
            // MOSS prompt 模板已显式拼装 audio_start / audio_end 等 token, 不需要 SP 的 <s>/</s>。
            string tokenizerPath = Path.Combine(ttsDir, manifest.ModelFiles.TokenizerModel);
            if (!File.Exists(tokenizerPath))
                throw new FileNotFoundException($"tokenizer.model 不存在: {tokenizerPath}", tokenizerPath);
            using (var tkStream = File.OpenRead(tokenizerPath))
            {
                _tokenizer = SentencePieceTokenizer.Create(
                    tkStream,
                    addBeginningOfSentence: false,
                    addEndOfSentence: false);
            }

            _engine = new InferenceEngine(_bundle);
            _generator = new AudioFrameGenerator(_engine);
            _streamingSession = new CodecStreamingDecodeSession(_bundle.CodecDecodeStep, codecMeta);
            _chineseTextNormalizer = CreateTextNormalizer(WetextLanguage.Chinese, options.EnableTextNormalization);
            _englishTextNormalizer = CreateTextNormalizer(WetextLanguage.English, options.EnableTextNormalization);
        }

        // SentencePiece 模型 normalizer_spec.name = nmt_nfkc, 对应 NFKC 兼容性归一化。
        // Microsoft.ML.Tokenizers 2.0.0 的 SentencePieceTokenizer 不会自动应用 normalizer_spec,
        // 导致全角标点 (，！？（）等) 走 byte-fallback (3 个 byte token), 模型听到这种输入会发出
        // 一个无意义音节。这里对所有送入 tokenizer 的文本提前做 NFKC, 把全角符号映射为半角,
        // 行为对齐 Python sentencepiece 的默认。
        //
        // 入口处一次性 NFKC 后存到 PreparedText，内部 Count/Encode 直接使用归一化文本。
        private static string NormalizeForTokenizer(string text)
            => string.IsNullOrEmpty(text) ? text : text.Normalize(System.Text.NormalizationForm.FormKC);

        // CountTokensNormalized / EncodeTextNormalized: 调用方已保证 text 是 NFKC 形式。
        private int CountTokensNormalized(string text) => _tokenizer.CountTokens(text);

        private int[] EncodeTextNormalized(string text) => _tokenizer.EncodeToIds(text).ToArray();

        /// <summary>
        /// 整段合成: 与 Python `synthesize` 1:1 对齐。
        /// 当 streaming=true 时, 每段内部走 streaming codec 解码, 但本 API 只在所有段合成完成后返回 PCM。
        /// 真正的逐 chunk 推流由 <see cref="SynthesizeStreaming"/> 完成。
        /// </summary>
        public SynthesizeResult Synthesize(
            string text,
            string? voice,
            string? promptAudioPath,
            MossNanoSampleMode? sampleMode,
            bool? doSample,
            bool streaming,
            int? maxNewFrames,
            int voiceCloneMaxTextTokens,
            int? seed,
            float? textTemperature, int? textTopK, float? textTopP,
            float? audioTemperature, int? audioTopK, float? audioTopP, float? audioRepetitionPenalty,
            CancellationToken ct)
        {
            var gen = ResolveGeneration(sampleMode, doSample, maxNewFrames, seed,
                textTemperature, textTopK, textTopP,
                audioTemperature, audioTopK, audioTopP, audioRepetitionPenalty);

            string? voiceLang = ResolveVoiceLanguage(voice, promptAudioPath);
            string preparedText = PrepareSynthesisText(text, voiceLang);
            var promptCodes = ResolvePromptAudioCodes(voice, promptAudioPath);
            var chunks = TextChunker.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens, CountTokensNormalized);

            int sampleRate = SampleRate;
            int channels = Channels;
            var allWaveforms = new List<float[]>(chunks.Count * 2);
            var allFrames = new List<int[]>(gen.MaxNewFrames * chunks.Count);

            for (int i = 0; i < chunks.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                string aligned = AlignChunkPunctuationForVoiceLanguage(chunks[i], voiceLang);
                var chunkGen = CapFramesForShortInput(gen, aligned);
                var chunkResult = SynthesizeSingleChunk(aligned, promptCodes, streaming, chunkGen, ct);
                allFrames.AddRange(chunkResult.Frames);
                if (chunkResult.InterleavedWaveform.Length > 0)
                    allWaveforms.Add(chunkResult.InterleavedWaveform);

                if (i < chunks.Count - 1)
                {
                    float pauseSeconds = TextChunker.EstimateInterChunkPauseSeconds(chunks[i]);
                    int pauseSamples = Math.Max(0, (int)Math.Round(sampleRate * pauseSeconds));
                    if (pauseSamples > 0)
                        allWaveforms.Add(new float[pauseSamples * channels]);
                }
            }

            float[] interleaved = ConcatWaveforms(allWaveforms);
            return new SynthesizeResult(interleaved, sampleRate, channels, allFrames, chunks);
        }

        /// <summary>
        /// 流式合成: 在 codec streaming decode 路径下边推理边产生 PCM 段。每段 PCM 通过 onChunk 回调上抛。
        /// 与 Python `synthesize_single_chunk` 中 `streaming=True` 路径对齐 + 多段拼接 + 段间静音填充。
        /// </summary>
        public void SynthesizeStreaming(
            string text,
            string? voice,
            string? promptAudioPath,
            MossNanoSampleMode? sampleMode,
            bool? doSample,
            int? maxNewFrames,
            int voiceCloneMaxTextTokens,
            int streamingChunkFrames,
            int? seed,
            float? textTemperature, int? textTopK, float? textTopP,
            float? audioTemperature, int? audioTopK, float? audioTopP, float? audioRepetitionPenalty,
            Action<float[], bool> onChunk,
            CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(onChunk);
            var gen = ResolveGeneration(sampleMode, doSample, maxNewFrames, seed,
                textTemperature, textTopK, textTopP,
                audioTemperature, audioTopK, audioTopP, audioRepetitionPenalty);

            string? voiceLang = ResolveVoiceLanguage(voice, promptAudioPath);
            string preparedText = PrepareSynthesisText(text, voiceLang);
            var promptCodes = ResolvePromptAudioCodes(voice, promptAudioPath);
            var chunks = TextChunker.SplitVoiceCloneText(preparedText, voiceCloneMaxTextTokens, CountTokensNormalized);

            int sampleRate = SampleRate;
            int channels = Channels;
            int decodeFlushBudget = Math.Max(1, streamingChunkFrames);

            for (int i = 0; i < chunks.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                var chunkText = chunks[i];
                string alignedChunkText = AlignChunkPunctuationForVoiceLanguage(chunkText, voiceLang);

                _streamingSession.Reset();
                var pendingFrames = new List<int[]>(decodeFlushBudget * 2);

                void Flush(bool force)
                {
                    if (pendingFrames.Count == 0) return;
                    if (!force && pendingFrames.Count < decodeFlushBudget) return;
                    int take = force ? pendingFrames.Count : Math.Min(pendingFrames.Count, decodeFlushBudget);
                    var slice = pendingFrames.GetRange(0, take);
                    pendingFrames.RemoveRange(0, take);
                    var decoded = _streamingSession.RunFrames(slice);
                    if (decoded is null || decoded.AudioLength <= 0) return;
                    onChunk(InterleaveChannels(decoded.Audio, decoded.Channels, decoded.TotalSamples, decoded.AudioLength), false);
                }

                int[] textTokenIds = EncodeTextNormalized(alignedChunkText);
                var requestRows = _engine.BuildVoiceCloneRequestRows(promptCodes, textTokenIds);
                var chunkGen = CapFramesForShortInput(gen, alignedChunkText);

                _generator.Generate(requestRows, chunkGen, (frames, step, frame) =>
                {
                    pendingFrames.Add((int[])frame.Clone());
                    Flush(false);
                }, ct);
                Flush(true);
                _streamingSession.Reset();

                bool isLastChunk = i == chunks.Count - 1;
                if (!isLastChunk)
                {
                    float pauseSeconds = TextChunker.EstimateInterChunkPauseSeconds(chunkText);
                    int pauseSamples = Math.Max(0, (int)Math.Round(sampleRate * pauseSeconds));
                    if (pauseSamples > 0)
                        onChunk(new float[pauseSamples * channels], false);
                }
            }

            onChunk(Array.Empty<float>(), true);
        }

        private struct SingleChunkResult
        {
            public float[] InterleavedWaveform;
            public List<int[]> Frames;
        }

        private SingleChunkResult SynthesizeSingleChunk(
            string text,
            IReadOnlyList<int[]> promptCodes,
            bool streaming,
            GenerationParams gen,
            CancellationToken ct)
        {
            int[] textTokenIds = EncodeTextNormalized(text);
            var requestRows = _engine.BuildVoiceCloneRequestRows(promptCodes, textTokenIds);

            if (!streaming)
            {
                var frames = _generator.Generate(requestRows, gen, null, ct);
                float[] interleaved = DecodeFullSafe(frames);
                return new SingleChunkResult { Frames = frames, InterleavedWaveform = interleaved };
            }

            var pendingFrames = new List<int[]>(8);
            var emittedSegments = new List<float[]>();
            _streamingSession.Reset();
            try
            {
                var generated = _generator.Generate(requestRows, gen, (frs, step, frame) =>
                {
                    pendingFrames.Add((int[])frame.Clone());
                    if (pendingFrames.Count >= 2) FlushBuffered();
                }, ct);
                FlushBuffered();
                float[] interleaved = ConcatWaveforms(emittedSegments);
                return new SingleChunkResult { Frames = generated, InterleavedWaveform = interleaved };
            }
            finally
            {
                _streamingSession.Reset();
            }

            void FlushBuffered()
            {
                if (pendingFrames.Count == 0) return;
                var slice = pendingFrames.GetRange(0, pendingFrames.Count);
                pendingFrames.Clear();
                var decoded = _streamingSession.RunFrames(slice);
                if (decoded is null || decoded.AudioLength <= 0) return;
                emittedSegments.Add(InterleaveChannels(decoded.Audio, decoded.Channels, decoded.TotalSamples, decoded.AudioLength));
            }
        }

        private float[] DecodeFullSafe(IReadOnlyList<int[]> frames)
        {
            try
            {
                var (channelArrays, audioLength) = _engine.DecodeFullAudio(frames);
                return InterleaveChannelArrays(channelArrays, audioLength);
            }
            catch
            {
                _streamingSession.Reset();
                int channels = Channels;
                var perChannel = new List<float>[channels];
                for (int c = 0; c < channels; c++) perChannel[c] = new List<float>();
                try
                {
                    for (int i = 0; i < frames.Count; i += 8)
                    {
                        int take = Math.Min(8, frames.Count - i);
                        var batch = new List<int[]>(take);
                        for (int j = 0; j < take; j++) batch.Add(frames[i + j]);
                        var decoded = _streamingSession.RunFrames(batch);
                        if (decoded is null || decoded.AudioLength <= 0) continue;
                        for (int c = 0; c < decoded.Channels && c < channels; c++)
                        {
                            for (int t = 0; t < decoded.AudioLength; t++)
                                perChannel[c].Add(decoded.Audio[c * decoded.TotalSamples + t]);
                        }
                    }
                }
                finally
                {
                    _streamingSession.Reset();
                }
                int len = perChannel[0].Count;
                var interleaved = new float[len * channels];
                for (int t = 0; t < len; t++)
                    for (int c = 0; c < channels; c++)
                        interleaved[t * channels + c] = ClampSample(c < perChannel.Length && t < perChannel[c].Count ? perChannel[c][t] : 0f);
                return interleaved;
            }
        }

        private IReadOnlyList<int[]> ResolvePromptAudioCodes(string? voice, string? promptAudioPath)
        {
            if (!string.IsNullOrEmpty(promptAudioPath))
                return _engine.EncodeReferenceAudio(promptAudioPath);

            string resolvedVoice = string.IsNullOrEmpty(voice) ? _bundle.Manifest.BuiltinVoices[0].Voice : voice;
            var match = _bundle.Manifest.BuiltinVoices.FirstOrDefault(v =>
                string.Equals(v.Voice, resolvedVoice, StringComparison.OrdinalIgnoreCase));
            if (match is null)
                throw new ArgumentException($"未找到内置音色: {resolvedVoice}", nameof(voice));

            var rows = new int[match.PromptAudioCodes.Count][];
            for (int i = 0; i < match.PromptAudioCodes.Count; i++) rows[i] = match.PromptAudioCodes[i].ToArray();
            return rows;
        }

        /// <summary>
        /// 返回音色 prompt 音频对应的语种 group: "zh" / "ja" / "ko" / "en" / null (未知/外部 prompt)。
        /// 用 manifest builtin_voices[].group 字段做最简关键字匹配。
        /// </summary>
        private string? ResolveVoiceLanguage(string? voice, string? promptAudioPath)
        {
            if (!string.IsNullOrEmpty(promptAudioPath)) return null;
            string resolvedVoice = string.IsNullOrEmpty(voice) ? _bundle.Manifest.BuiltinVoices[0].Voice : voice;
            var match = _bundle.Manifest.BuiltinVoices.FirstOrDefault(v =>
                string.Equals(v.Voice, resolvedVoice, StringComparison.OrdinalIgnoreCase));
            string g = match?.Group ?? string.Empty;
            if (g.Contains("Chinese", StringComparison.OrdinalIgnoreCase)) return "zh";
            if (g.Contains("Japanese", StringComparison.OrdinalIgnoreCase)) return "ja";
            if (g.Contains("Korean", StringComparison.OrdinalIgnoreCase)) return "ko";
            if (g.Contains("English", StringComparison.OrdinalIgnoreCase)) return "en";
            return null;
        }

        /// <summary>
        /// 当 prompt 音频是 CN/JP/KO 但当前 chunk 是纯西文 (无 CJK) 时, 把句末的西文终止符替换为对应全角符号,
        /// 让模型从 "中文播音员" 视角看到熟悉的句末信号并采到 audio_end_token。
        /// 不改变 chunk 中间的标点 (避免影响节奏与 SP token 切分)。
        /// </summary>
        private static string AlignChunkPunctuationForVoiceLanguage(string chunk, string? voiceLang)
        {
            if (string.IsNullOrEmpty(chunk) || voiceLang is null or "en") return chunk;
            if (TextChunker.ContainsCjk(chunk)) return chunk;

            string trimmed = chunk.TrimEnd();
            if (trimmed.Length == 0) return chunk;
            char last = trimmed[^1];
            char? replacement = last switch
            {
                '.' => '。',
                '!' => '！',
                '?' => '？',
                ';' => '；',
                ',' => '，',
                ':' => '：',
                _ => null,
            };
            if (replacement is null) return chunk;
            return trimmed[..^1] + replacement.Value;
        }

        // 每秒 audio frame 数 = 采样率 / codec hop (downsample_rate)。
        // MOSS-Audio-Tokenizer-Nano: 48000 / 3840 = 12.5 fps, 即每帧 80ms。
        private double FramesPerSecond => SampleRate / (double)Math.Max(1, _bundle.CodecMeta.CodecConfig.DownsampleRate);

        private static bool IsCjkChar(char c)
            => c is >= '一' and <= '鿿'
                or >= '㐀' and <= '䶿'
                or >= '぀' and <= 'ヿ'
                or >= '가' and <= '힯';

        /// <summary>
        /// 统计 chunk 的长度信号:
        ///   - cjkChars: CJK 字符数;
        ///   - letterSegments: 连续字母段的个数 ("hello world" → 2, "oh" → 1, "2023.05.06" → 0);
        ///   - letterChars: 字母总字符数 (不含数字/标点)。
        /// 数字与标点既不计入字母, 也不打断"段"以外的统计 —— 但字母段以非字母字符为分隔。
        /// </summary>
        private static (int cjkChars, int letterSegments, int letterChars) CountChunkLengthSignals(string chunk)
        {
            int cjk = 0, segments = 0, letters = 0;
            bool inLetterRun = false;
            foreach (char c in chunk)
            {
                if (IsCjkChar(c)) { cjk++; inLetterRun = false; continue; }
                if (char.IsLetter(c))
                {
                    if (!inLetterRun) { segments++; inLetterRun = true; }
                    letters++;
                }
                else
                {
                    inLetterRun = false;
                }
            }
            return (cjk, segments, letters);
        }

        private const string SpokenNumericChars = "零〇一二两三四五六七八九十百千万亿兆点负正分之百分比半";

        /// <summary>
        /// 识别短数字表达式，包括原始的 "2.6" 和 TN 后的 "二点六"。
        /// 返回近似朗读单元数，用于估算安全的最大生成时长。
        /// </summary>
        private static bool TryCountShortNumericUnits(string chunk, out int unitCount)
        {
            unitCount = 0;
            string value = chunk.Trim();
            if (value.Length == 0) return false;
            if (value[^1] is '.' or '!' or '?' or '。' or '！' or '？' or ';' or '；')
                value = value[..^1].TrimEnd();

            int digitCount = 0;
            bool rawNumeric = true;
            foreach (char c in value)
            {
                if (char.IsDigit(c))
                {
                    digitCount++;
                    unitCount++;
                    continue;
                }
                if (char.IsWhiteSpace(c))
                    continue;
                if (char.IsPunctuation(c) || char.IsSymbol(c))
                {
                    unitCount++;
                    continue;
                }
                rawNumeric = false;
                break;
            }
            if (rawNumeric && digitCount is > 0 and <= 6)
                return true;

            unitCount = 0;
            foreach (char c in value)
            {
                if (SpokenNumericChars.Contains(c, StringComparison.Ordinal))
                {
                    unitCount++;
                    continue;
                }
                if (char.IsWhiteSpace(c) || char.IsPunctuation(c) || char.IsSymbol(c))
                    continue;
                return false;
            }
            return unitCount is > 0 and <= 12;
        }

        /// <summary>
        /// 短输入兜底: 限制 MaxNewFrames, 防止模型一直不产生 audio_end。
        /// 触发规则:
        ///   - chunk 含 CJK: 当 CJK 字数 ≤ 2;
        ///   - chunk 不含 CJK: 仅当全文只有一段连续字母, 且字母总数 ≤ 3 (覆盖 "oh"/"yes" 这类短英文)。
        ///   - 短数字表达式: 原始数字位数 ≤ 6，或 TN 后的数字朗读单元 ≤ 12。
        ///     超过上述长度的日期、电话号码等不会误触发。
        /// 仅 clone 后修改, 不改原 gen。只有当 cap 真正小于原 MaxNewFrames (即发生了截断) 时才记录日志。
        /// </summary>
        private GenerationParams CapFramesForShortInput(GenerationParams gen, string chunk)
        {
            if (string.IsNullOrWhiteSpace(chunk)) return gen;
            var (cjk, letterSegments, letterChars) = CountChunkLengthSignals(chunk);
            bool shortNumeric = TryCountShortNumericUnits(chunk, out int numericUnits);
            bool ultraShort = cjk > 0
                ? cjk <= 2
                : letterSegments == 1 && letterChars <= 3;
            if (!ultraShort && !shortNumeric) return gen;

            double capSeconds = shortNumeric
                ? Math.Clamp(numericUnits * 0.9, 1.2, 10.8)
                : 1.0;
            int cap = Math.Max(1, (int)Math.Ceiling(FramesPerSecond * capSeconds));
            // 仅当原 MaxNewFrames 严格大于 cap 时才截断并打日志; 否则保持不变, 不打日志。
            if (gen.MaxNewFrames <= cap) return gen;

            var clone = gen.Clone();
            clone.MaxNewFrames = cap;
            if (_logger.IsEnabled(LogLevel.Information))
            {
                _logger.LogInformation(
                    "MOSS-TTS-Nano 检测到短输入 (chunk=\"{Chunk}\", cjk={Cjk}, letterSegments={Segments}, letterChars={Letters}, numericUnits={NumericUnits}), MaxNewFrames {OrigFrames}→{CapFrames} (≈{CapSeconds:F1}s)。",
                    chunk.Trim(), cjk, letterSegments, letterChars, numericUnits,
                    gen.MaxNewFrames, clone.MaxNewFrames, capSeconds);
            }
            return clone;
        }

        private GenerationParams ResolveGeneration(
            MossNanoSampleMode? sampleMode, bool? doSample,
            int? maxNewFrames, int? seed,
            float? textTemperature, int? textTopK, float? textTopP,
            float? audioTemperature, int? audioTopK, float? audioTopP, float? audioRepetitionPenalty)
        {
            var defaults = _bundle.Manifest.GenerationDefaults;

            // Sample mode 归一化
            MossNanoSampleMode normalized;
            if (sampleMode.HasValue)
                normalized = (doSample == false) ? MossNanoSampleMode.Greedy : sampleMode.Value;
            else
                normalized = SampleModeUtil.Normalize(defaults.SampleMode, doSample ?? defaults.DoSample);

            var gen = GenerationParams.FromManifest(defaults, normalized);
            if (maxNewFrames.HasValue) gen.MaxNewFrames = maxNewFrames.Value;
            if (textTemperature.HasValue) gen.TextTemperature = textTemperature.Value;
            if (textTopK.HasValue) gen.TextTopK = textTopK.Value;
            if (textTopP.HasValue) gen.TextTopP = textTopP.Value;
            if (audioTemperature.HasValue) gen.AudioTemperature = audioTemperature.Value;
            if (audioTopK.HasValue) gen.AudioTopK = audioTopK.Value;
            if (audioTopP.HasValue) gen.AudioTopP = audioTopP.Value;
            if (audioRepetitionPenalty.HasValue) gen.AudioRepetitionPenalty = audioRepetitionPenalty.Value;

            if (seed.HasValue) _engine.ResetRandom(seed.Value);
            return gen;
        }

        private string PrepareSynthesisText(string text, string? voiceLanguage)
        {
            string prepared = PrepareText(text);
            prepared = NormalizeText(prepared, voiceLanguage);
            return NormalizeForTokenizer(prepared);
        }

        private WetextProcessor? CreateTextNormalizer(WetextLanguage language, bool enabled)
        {
            if (!enabled) return null;

            try
            {
                return new WetextProcessor(language, WetextMode.TextNormalization);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "MOSS-TTS-Nano {Language} text normalizer initialization failed; matching text will be passed through.",
                    language);
                return null;
            }
        }

        private string NormalizeText(string text, string? voiceLanguage)
        {
            // WeTextProcessing.NET 仅提供中英文模型，日语和韩语保持原样。
            if (voiceLanguage is "ja" or "ko" || ContainsJapaneseOrKorean(text)) return text;

            bool useChinese = ContainsChineseChar(text)
                || (!ContainsLatinLetter(text) && voiceLanguage == "zh");
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
                _logger.LogDebug(ex, "MOSS-TTS-Nano text normalization failed; using original text.");
                return text;
            }
        }

        private static bool ContainsChineseChar(string text)
        {
            foreach (char c in text)
            {
                if (c is >= '一' and <= '鿿' or >= '㐀' and <= '䶿') return true;
            }
            return false;
        }

        private static bool ContainsLatinLetter(string text)
        {
            foreach (char c in text)
            {
                if (c is >= 'A' and <= 'Z' or >= 'a' and <= 'z') return true;
            }
            return false;
        }

        private static bool ContainsJapaneseOrKorean(string text)
        {
            foreach (char c in text)
            {
                if (c is >= '぀' and <= 'ヿ' or >= '가' and <= '힯') return true;
            }
            return false;
        }

        private static string PrepareText(string text)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(text);
            // WeTextProcessing 前先规整换行、全角空格并去掉零宽字符。
            string s = text.Replace("\r\n", "\n").Replace("\r", "\n").Replace('　', ' ');
            s = StripZeroWidth(s);
            return s.Trim();
        }

        private static string StripZeroWidth(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            var sb = new System.Text.StringBuilder(s.Length);
            foreach (char c in s)
            {
                if (c == '​' || c == '‌' || c == '‍' || c == '﻿') continue;
                sb.Append(c);
            }
            return sb.ToString();
        }

        /// <summary>
        /// channel-major [c, t] (前 audioLength 个样本有效) → frame-major (interleaved) [s0c0, s0c1, ...]
        /// 输出会 clamp 到 [-1, 1]: codec 偶尔会输出 |x|&gt;1 的样本 (Python 也有),
        /// 不 clamp 会让 PortAudio/16-bit WAV 转换时整数溢出回卷, 产生明显爆音。
        /// </summary>
        private static float[] InterleaveChannels(float[] channelMajor, int channels, int totalSamples, int audioLength)
        {
            var result = new float[audioLength * channels];
            for (int t = 0; t < audioLength; t++)
            {
                for (int c = 0; c < channels; c++)
                    result[t * channels + c] = ClampSample(channelMajor[c * totalSamples + t]);
            }
            return result;
        }

        private static float[] InterleaveChannelArrays(float[][] channelArrays, int audioLength)
        {
            int channels = channelArrays.Length;
            if (channels == 0 || audioLength == 0) return Array.Empty<float>();
            var result = new float[audioLength * channels];
            for (int t = 0; t < audioLength; t++)
                for (int c = 0; c < channels; c++)
                    result[t * channels + c] = ClampSample(channelArrays[c][t]);
            return result;
        }

        private static float ClampSample(float v)
        {
            if (v > 1f) return 1f;
            if (v < -1f) return -1f;
            return v;
        }

        private static float[] ConcatWaveforms(List<float[]> waveforms)
        {
            int total = 0;
            foreach (var w in waveforms) total += w.Length;
            if (total == 0) return Array.Empty<float>();
            var result = new float[total];
            int o = 0;
            foreach (var w in waveforms) { Array.Copy(w, 0, result, o, w.Length); o += w.Length; }
            return result;
        }

        private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true };

        private static T LoadJson<T>(string path) where T : new()
        {
            using var stream = File.OpenRead(path);
            var value = JsonSerializer.Deserialize<T>(stream, JsonOptions);
            return value ?? new T();
        }

        private static string ResolveRelativeFile(string baseDir, string relative)
        {
            if (Path.IsPathRooted(relative))
                return Path.GetFullPath(relative);
            return Path.GetFullPath(Path.Combine(baseDir, relative));
        }

        public void Warmup(CancellationToken ct = default)
        {
            try
            {
                ct.ThrowIfCancellationRequested();
                var voice = _bundle.Manifest.BuiltinVoices.FirstOrDefault();
                if (voice is null) return;

                var promptCodes = new int[voice.PromptAudioCodes.Count][];
                for (int i = 0; i < promptCodes.Length; i++)
                    promptCodes[i] = voice.PromptAudioCodes[i].ToArray();

                int[] textTokens = EncodeTextNormalized(NormalizeForTokenizer("a"));
                var requestRows = _engine.BuildVoiceCloneRequestRows(promptCodes, textTokens);

                var gen = GenerationParams.FromManifest(_bundle.Manifest.GenerationDefaults,
                    SampleModeUtil.Normalize(_bundle.Manifest.GenerationDefaults.SampleMode,
                        _bundle.Manifest.GenerationDefaults.DoSample));
                gen.MaxNewFrames = 2;

                // 预热阶段会消耗 RNG; 跑完恢复初始种子, 让真实请求从干净状态开始,
                // 避免 "服务启动后第一次请求" 的输出受预热步数偏移影响。
                _engine.ResetRandom(1234);
                try
                {
                    _generator.Generate(requestRows, gen, null, ct);
                    ct.ThrowIfCancellationRequested();

                    var dummyFrame = new int[_bundle.CodecMeta.CodecConfig.NumQuantizers];
                    var dummyFrames = new List<int[]> { dummyFrame };
                    _engine.DecodeFullAudio(dummyFrames);
                    ct.ThrowIfCancellationRequested();
                    _streamingSession.Reset();
                    _streamingSession.RunFrames(dummyFrames);
                    _streamingSession.Reset();
                }
                finally
                {
                    _engine.ResetRandom(1234);
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                throw;
            }
            catch
            {
                // 预热失败不影响真实请求, 上层会自然报错。
            }
        }

        public void Dispose()
        {
            _streamingSession.Dispose();
            _engine.Dispose();
            _bundle.Dispose();
            _chineseTextNormalizer?.Dispose();
            _englishTextNormalizer?.Dispose();
            if (_ownsTtsSessionOptions) _ownedTtsSessionOptions?.Dispose();
        }
    }

    /// <summary>
    /// 完整一次 voice-clone 合成的结果。
    /// </summary>
    internal sealed class SynthesizeResult
    {
        public SynthesizeResult(float[] interleavedWaveform, int sampleRate, int channels,
            IReadOnlyList<int[]> generatedFrames, IReadOnlyList<string> textChunks)
        {
            InterleavedWaveform = interleavedWaveform;
            SampleRate = sampleRate;
            Channels = channels;
            GeneratedFrames = generatedFrames;
            TextChunks = textChunks;
        }

        public float[] InterleavedWaveform { get; }
        public int SampleRate { get; }
        public int Channels { get; }
        public IReadOnlyList<int[]> GeneratedFrames { get; }
        public IReadOnlyList<string> TextChunks { get; }
    }
}
