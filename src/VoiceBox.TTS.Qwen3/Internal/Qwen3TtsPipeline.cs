using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Qwen3-TTS pipeline: tokenize → build prompt embeddings → talker autoregressive loop
    /// (each frame: talker step → sample group0 → code-predictor 15 steps → next input) → codec decode.
    /// Faithful port of the validated Python reference (_engine.py): prompt parity 141 dB,
    /// greedy fp32 codes byte-exact vs PyTorch.
    /// </summary>
    internal sealed class Qwen3TtsPipeline : IDisposable
    {
        private readonly Qwen3TTSOptions _options;
        private readonly ILogger? _logger;
        private readonly Qwen3ModelConfig _cfg;
        private readonly EmbeddingStore _store;
        private readonly Qwen3Tokenizer _tokenizer;
        private readonly TalkerDecoder _talker;
        private readonly CodePredictor _cp;
        private readonly CodecDecoder _codec;
        private readonly SessionOptions _so;
        private readonly SessionOptions _codecSo;
        private readonly IReadOnlyList<string> _speakers;
        private readonly int _H;
        private readonly bool _supportsInstruct;
        private int _seedCounter;
        private static readonly bool Diag = Environment.GetEnvironmentVariable("QWEN3TTS_DIAG") == "1";
        private static readonly bool Profile = Environment.GetEnvironmentVariable("QWEN3TTS_PROFILE") == "1";

        public int SampleRate => _cfg.SampleRate;
        public int Channels => 1;
        public string DefaultVoiceName { get; }

        public Qwen3TtsPipeline(Qwen3TTSOptions options, ILogger? logger)
        {
            _options = options;
            _logger = logger;
            if (string.IsNullOrWhiteSpace(options.ModelPath) || !Directory.Exists(options.ModelPath))
                throw new DirectoryNotFoundException($"Qwen3-TTS 模型目录不存在: {options.ModelPath}");

            _cfg = Qwen3ModelConfig.Load(options.ModelPath);
            _H = _cfg.Talker.HiddenSize;
            _supportsInstruct = !string.Equals(_cfg.TtsModelSize, "0b6", StringComparison.OrdinalIgnoreCase);

            // Avoid ORT's default EXHAUSTIVE cuDNN search (pathological on dynamic shapes).
            if (options.ExecutionProvider == OnnxExecutionProvider.Cuda)
                options.CudaTuning ??= new CudaExecutionTuning();

            _store = new EmbeddingStore(options.ModelPath, _cfg);
            _tokenizer = new Qwen3Tokenizer(Path.Combine(options.ModelPath, "tokenizer"));

            // Build native (disposable) resources into locals; assign fields only once ALL succeed,
            // so a partway failure (missing onnx / ORT init) doesn't leak earlier sessions.
            SessionOptions? so = null, codecSo = null;
            TalkerDecoder? talker = null;
            CodePredictor? cp = null;
            CodecDecoder? codec = null;
            try
            {
                so = OnnxSessionFactory.Create(options);                       // talker / code_predictor (matmul-heavy)
                // Codec is fp32 either way. CUDA is the default when the main EP is CUDA; CPU remains
                // available for low-VRAM/debug fallback.
                bool codecOnCuda = options.ExecutionProvider == OnnxExecutionProvider.Cuda;
                codecSo = codecOnCuda ? OnnxSessionFactory.Create(options) : OnnxSessionFactory.CreateCpu(options.ThreadsNum);
                // Talker keeps its growing KV in VRAM via IO-binding on CUDA (no per-frame KV copy).
                bool talkerIoBinding = options.ExecutionProvider == OnnxExecutionProvider.Cuda;
                talker = new TalkerDecoder(Path.Combine(options.ModelPath, "talker.onnx"), so, _cfg, talkerIoBinding, options.GpuDeviceId);
                cp = new CodePredictor(options.ModelPath, so, _cfg, _store);
                codec = new CodecDecoder(Path.Combine(options.ModelPath, "codec_decoder.onnx"), codecSo, _cfg);
            }
            catch
            {
                codec?.Dispose(); cp?.Dispose(); talker?.Dispose(); codecSo?.Dispose(); so?.Dispose();
                throw;
            }
            _so = so; _codecSo = codecSo; _talker = talker; _cp = cp; _codec = codec;

            _speakers = _cfg.SpeakerId.Keys.ToList();
            if (_speakers.Count == 0)
                throw new InvalidDataException("config.json 中没有可用的 speaker。");
            DefaultVoiceName = !string.IsNullOrWhiteSpace(options.DefaultVoice) ? options.DefaultVoice! : _speakers[0];
            TryWarmup();
        }

        public IReadOnlyList<TtsVoice> ListVoices()
        {
            var list = new List<TtsVoice>(_speakers.Count);
            foreach (var s in _speakers)
            {
                _cfg.SpeakerIsDialect.TryGetValue(s, out string? dialect);
                list.Add(new TtsVoice { Name = s, Description = dialect is null ? null : $"dialect: {dialect}" });
            }
            return list;
        }

        // ---- whole / streaming entry points ----

        public float[] SynthesizeWhole(string text, TtsSynthesisOptions? opts, out string voice, CancellationToken ct)
        {
            var (prompt, attnLen, pad, gp, rng) = Prepare(text, opts, out voice);
            var frames = new List<int[]>();
            foreach (var f in GenerateFrames(prompt, attnLen, pad, gp, rng, ct)) frames.Add(f);
            ct.ThrowIfCancellationRequested();
            return _codec.Decode(frames);
        }

        /// <summary>Test hook: generate the raw codec frames [T][G] (for parity vs Python golden).</summary>
        internal List<int[]> GenerateCodes(string text, TtsSynthesisOptions? opts, CancellationToken ct)
        {
            var (prompt, attnLen, pad, gp, rng) = Prepare(text, opts, out _);
            var frames = new List<int[]>();
            foreach (var f in GenerateFrames(prompt, attnLen, pad, gp, rng, ct)) frames.Add(f);
            return frames;
        }

        /// <summary>Test hook: decode pre-generated frames to a waveform.</summary>
        internal float[] DecodeFrames(IReadOnlyList<int[]> frames) => _codec.Decode(frames);

        /// <summary>Test hook: decode pre-generated frames with the streaming chunk scheduler.</summary>
        internal float[] DecodeFramesStreamedForTest(IReadOnlyList<int[]> frames, int[] schedule, int leftCtx)
        {
            if (frames.Count == 0) return [];
            if (schedule.Length == 0) schedule = [1, 4, 8, 16, 32, 48];

            var output = new List<float[]>();
            int total = 0, decodedFrames = 0, blockIdx = 0;
            while (decodedFrames < frames.Count)
            {
                int blockSize = schedule[Math.Min(blockIdx, schedule.Length - 1)];
                int end = Math.Min(frames.Count, decodedFrames + blockSize);
                var (window, drop) = SnapshotWindowForRange(frames, decodedFrames, end, leftCtx);
                decodedFrames = end;
                float[] wav = _codec.DecodeFramesContiguous(window);
                float[] outp = wav.AsSpan(drop).ToArray();
                output.Add(outp);
                total += outp.Length;
                blockIdx++;
            }
            var merged = new float[total];
            int offset = 0;
            foreach (var part in output)
            {
                part.CopyTo(merged, offset);
                offset += part.Length;
            }
            return merged;
        }

        public IEnumerable<float[]> SynthesizeStream(string text, TtsSynthesisOptions? opts, CancellationToken ct)
        {
            var (prompt, attnLen, pad, gp, rng) = Prepare(text, opts, out _);
            int[] schedule = _options.StreamChunkSchedule is { Length: > 0 } ? _options.StreamChunkSchedule : [1, 4, 8, 16, 32, 48];
            int leftCtx = Math.Max(0, _options.StreamLeftContextFrames);

            // Pipeline generation and codec decode run on two tasks: the generator pushes decode-windows
            // to a small bounded queue, this consumer runs the codec. CPU codec overlaps the GPU AR loop;
            // CUDA codec keeps chunk delivery asynchronous.
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            var token = cts.Token;
            var windows = new System.Collections.Concurrent.BlockingCollection<(List<int[]> window, int drop)>(2);
            var swCodec = new System.Diagnostics.Stopwatch();

            void Producer()
            {
                try
                {
                    var frames = new List<int[]>();
                    int decodedFrames = 0, blockIdx = 0;
                    foreach (var f in GenerateFrames(prompt, attnLen, pad, gp, rng, token))
                    {
                        frames.Add(f);
                        int blockSize = schedule[Math.Min(blockIdx, schedule.Length - 1)];
                        if (frames.Count - decodedFrames >= blockSize)
                        {
                            windows.Add(SnapshotWindow(frames, ref decodedFrames, leftCtx), token);
                            blockIdx++;
                        }
                    }
                    if (decodedFrames < frames.Count)
                        windows.Add(SnapshotWindow(frames, ref decodedFrames, leftCtx), token);
                }
                finally { windows.CompleteAdding(); }
            }

            var genTask = Task.Run(Producer, token);

            try
            {
                foreach (var (window, drop) in windows.GetConsumingEnumerable(token))
                {
                    swCodec.Start();
                    float[] wav = _codec.DecodeFramesContiguous(window);
                    float[] outp = wav.AsSpan(drop).ToArray();
                    swCodec.Stop();
                    yield return outp;
                }
            }
            finally
            {
                cts.Cancel();                       // unblock the generator if the consumer stopped early
                try
                {
                    try { genTask.GetAwaiter().GetResult(); }   // surface a real generation error; swallow cancellation
                    catch (OperationCanceledException) { }
                }
                finally
                {
                    windows.Dispose();
                    if (Environment.GetEnvironmentVariable("QWEN3TTS_PROFILE") == "1")
                        Console.Error.WriteLine($"[qwen3 profile] codec(async)={swCodec.ElapsedMilliseconds}ms");
                }
            }
        }

        /// <summary>Snapshot frames [decodedFrames-leftCtx, end) as a COPY (safe to decode off-thread)
        /// and advance <paramref name="decodedFrames"/>. Returns the window and the left-context sample
        /// count to drop from the decoded waveform.</summary>
        private (List<int[]> window, int drop) SnapshotWindow(List<int[]> frames, ref int decodedFrames, int leftCtx)
        {
            int end = frames.Count;
            int ctx = decodedFrames - leftCtx > 0 ? leftCtx : decodedFrames;
            int from = decodedFrames - ctx;
            var window = frames.GetRange(from, end - from);   // copies the element refs -> safe for the codec thread
            decodedFrames = end;
            return (window, ctx * _cfg.SamplesPerFrame);
        }

        private (List<int[]> window, int drop) SnapshotWindowForRange(
            IReadOnlyList<int[]> frames, int decodedFrames, int end, int leftCtx)
        {
            int ctx = decodedFrames - leftCtx > 0 ? leftCtx : decodedFrames;
            int from = decodedFrames - ctx;
            var window = new List<int[]>(end - from);
            for (int i = from; i < end; i++) window.Add(frames[i]);
            return (window, ctx * _cfg.SamplesPerFrame);
        }

        // ---- generation ----

        private sealed record GenParams(
            bool DoSample, float Temp, int TopK, float TopP, float RepPenalty,
            bool SubDoSample, float SubTemp, int SubTopK, float SubTopP, int MaxFrames);

        private (float[] prompt, int attnLen, float[] pad, GenParams gp, Random rng) Prepare(
            string text, TtsSynthesisOptions? opts, out string voice)
        {
            var swPrepare = Profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            var swStage = Profile ? System.Diagnostics.Stopwatch.StartNew() : null;
            ArgumentException.ThrowIfNullOrEmpty(text);
            voice = ResolveVoice(opts?.Voice);
            string language = opts?.Language ?? _options.DefaultLanguage ?? "auto";
            if (!string.Equals(language, "auto", StringComparison.OrdinalIgnoreCase)
                && !_cfg.CodecLanguageId.ContainsKey(language))
                throw new ArgumentException(
                    $"未知语言 '{language}'。可用: auto, {string.Join(", ", _cfg.CodecLanguageId.Keys)}");
            string? instruct = _supportsInstruct ? opts?.Instruct : null;

            int[] textIds = _tokenizer.EncodeAssistant(text);
            int[]? instructIds = string.IsNullOrEmpty(instruct) ? null : _tokenizer.EncodeInstruct(instruct!);
            long tokenizeMs = swStage?.ElapsedMilliseconds ?? 0;
            swStage?.Restart();
            var (prompt, pad) = PromptBuilder.Build(_store, _cfg, textIds, voice, language, instructIds);
            long promptMs = swStage?.ElapsedMilliseconds ?? 0;
            int attnLen = prompt.Length / _H;

            var g = _cfg.Generation;
            var gp = new GenParams(
                DoSample: opts?.DoSample ?? _options.DoSample ?? g.DoSample,
                Temp: opts?.AudioTemperature ?? _options.Temperature ?? g.Temperature,
                TopK: opts?.AudioTopK ?? _options.TopK ?? g.TopK,
                TopP: opts?.AudioTopP ?? _options.TopP ?? g.TopP,
                RepPenalty: opts?.AudioRepetitionPenalty ?? _options.RepetitionPenalty ?? g.RepetitionPenalty,
                SubDoSample: _options.SubtalkerDoSample ?? g.SubtalkerDoSample,
                SubTemp: _options.SubtalkerTemperature ?? g.SubtalkerTemperature,
                SubTopK: _options.SubtalkerTopK ?? g.SubtalkerTopK,
                SubTopP: _options.SubtalkerTopP ?? g.SubtalkerTopP,
                MaxFrames: opts?.MaxNewFrames ?? _options.MaxNewFrames ?? g.MaxNewTokens);

            int? seed = opts?.Seed ?? _options.DefaultSeed;
            var rng = seed.HasValue ? new Random(seed.Value) : new Random(unchecked(Environment.TickCount + _seedCounter++));
            if (Profile)
                Console.Error.WriteLine(
                    $"[qwen3 profile] prepare total={swPrepare!.ElapsedMilliseconds}ms  " +
                    $"tokenize={tokenizeMs}ms  prompt={promptMs}ms  " +
                    $"textTokens={textIds.Length} promptLen={attnLen} projectedText={_store.UsesProjectedTextEmbeddings}");
            return (prompt, attnLen, pad, gp, rng);
        }

        private IEnumerable<int[]> GenerateFrames(
            float[] prompt, int attnLen, float[] pad, GenParams gp, Random rng, CancellationToken ct)
        {
            var swTalker = new System.Diagnostics.Stopwatch();
            var swCp = new System.Diagnostics.Stopwatch();
            int nframes = 0;

            _talker.Reset();
            int promptLen = prompt.Length / _H;
            try
            {
                swTalker.Start();
                var (logits0, hidden0) = _talker.Step(prompt, attnLen, 0, promptLen);
                swTalker.Stop();
                long prefillMs = swTalker.ElapsedMilliseconds;
                var history = new HashSet<int>();   // unique generated group0 tokens (for repetition penalty)
                int group0 = SampleTalker(logits0, history, 0, gp, rng);
                history.Add(group0);
                float[] pastHidden = hidden0;
                int total = attnLen;
                int eos = _cfg.SpecialTokens.CodecEos;
                int G = _cfg.NumCodeGroups;

                // The fused code_predictor samples its 15 fine groups in-graph (temperature is passed;
                // top_k is baked from generation_config — SubTopK/SubTopP overrides are not honored by
                // the fused path). Supply one uniform per group when sampling so the RNG stays seedable.
                var subRandoms = new float[_cp.RandomCount];

                for (int frame = 0; frame < gp.MaxFrames; frame++)
                {
                    ct.ThrowIfCancellationRequested();
                    if (group0 == eos) break;

                    if (gp.SubDoSample)
                        for (int i = 0; i < subRandoms.Length; i++) subRandoms[i] = (float)rng.NextDouble();

                    swCp.Start();
                    var (g, nextSum) = _cp.GenerateFrame(pastHidden, group0, gp.SubDoSample, gp.SubTemp, subRandoms);
                    swCp.Stop();
                    nframes++;
                    var codes = new int[G];
                    codes[0] = group0;
                    Array.Copy(g, 0, codes, 1, G - 1);
                    yield return codes;

                    // next talker input = (codec(group0) + Σ cp_emb[gs][code_gs]) + tts_pad;
                    // the fused cp returns that 16-code embedding sum as nextSum.
                    var nxt = new float[_H];
                    for (int h = 0; h < _H; h++) nxt[h] = nextSum[h] + pad[h];

                    total += 1;
                    swTalker.Start();
                    var (logits, hidden) = _talker.Step(nxt, total, total - 1, 1);
                    swTalker.Stop();
                    group0 = SampleTalker(logits, history, frame + 1, gp, rng);
                    history.Add(group0);
                    pastHidden = hidden;
                }

                if (Diag || Profile)
                {
                    long decodeTalker = swTalker.ElapsedMilliseconds - prefillMs;
                    Console.Error.WriteLine(
                        $"[qwen3 profile] frames={nframes}  prefill={prefillMs}ms  " +
                        $"talker-decode={decodeTalker}ms ({(nframes > 0 ? decodeTalker / (double)nframes : 0):F2}/frame)  " +
                        $"cp={swCp.ElapsedMilliseconds}ms ({(nframes > 0 ? swCp.ElapsedMilliseconds / (double)nframes : 0):F2}/frame)");
                }
            }
            finally
            {
                // 合成结束 (正常 / EOS / 取消 / 提前 break): 释放 talker 每次生成的 KV 状态,
                // 使 IO-binding 的整段 present_k/present_v 不在空闲期常驻显存。下次 GenerateFrames
                // 起始的 _talker.Reset() 会重建 binding/empty-past。
                _talker.EndGeneration();
            }
        }

        /// <summary>Process talker group0 logits (repetition penalty, min-new-tokens, suppress) then sample.</summary>
        private int SampleTalker(float[] logits, HashSet<int> history, int emitted, GenParams gp, Random rng)
        {
            var g = _cfg.Generation;
            if (Diag && emitted < 6)
            {
                int nan = 0, inf = 0; float mx = float.NegativeInfinity; int amax = 0;
                for (int i = 0; i < logits.Length; i++)
                {
                    if (float.IsNaN(logits[i])) nan++;
                    else if (float.IsInfinity(logits[i])) inf++;
                    else if (logits[i] > mx) { mx = logits[i]; amax = i; }
                }
                Console.Error.WriteLine($"[diag] frame={emitted} rawArgmax={amax} rawMax={mx:F3} " +
                    $"eosLogit={logits[_cfg.SpecialTokens.CodecEos]:F3} nan={nan} inf={inf}");
            }
            // repetition penalty over generated group0 history — ONCE per unique token (matches HF/_engine.py)
            if (Math.Abs(gp.RepPenalty - 1f) > 1e-6f)
            {
                foreach (int tid in history)
                {
                    float s = logits[tid];
                    logits[tid] = s < 0 ? s * gp.RepPenalty : s / gp.RepPenalty;
                }
            }
            if (emitted < g.MinNewTokens) logits[_cfg.SpecialTokens.CodecEos] = float.NegativeInfinity;
            var sup = _cfg.SuppressTokens;
            for (int t = sup.RangeStart; t < sup.RangeEnd; t++)
                if (Array.IndexOf(sup.Keep, t) < 0) logits[t] = float.NegativeInfinity;
            return Sampler.Sample(logits, gp.DoSample, gp.Temp, gp.TopK, gp.TopP, rng);
        }

        private string ResolveVoice(string? requested)
        {
            string v = string.IsNullOrWhiteSpace(requested) ? DefaultVoiceName : requested!;
            if (!_cfg.SpeakerId.ContainsKey(v))
                throw new ArgumentException($"未知音色 '{v}'。可用: {string.Join(", ", _speakers)}");
            return v;
        }

        private void TryWarmup()
        {
            try
            {
                using var ctsrc = new CancellationTokenSource(TimeSpan.FromSeconds(60));
                var opts = new TtsSynthesisOptions { DoSample = false, MaxNewFrames = 2 };
                _ = SynthesizeWhole("。", opts, out _, ctsrc.Token);
            }
            catch (Exception ex) { _logger?.LogWarning(ex, "Qwen3-TTS 预热失败 (忽略)。"); }
        }

        public void Dispose()
        {
            _talker.Dispose();
            _cp.Dispose();
            _codec.Dispose();
            _so.Dispose();
            _codecSo.Dispose();
        }
    }
}
