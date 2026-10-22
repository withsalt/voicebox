using System;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.ASR.Qwen3.Streaming;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// One-shot transcription pipeline: waveform → mel → encoder → decoder → text.
    /// Owns the heavyweight ONNX sessions; designed to be created once and shared
    /// across sessions via a recognition gate.
    /// </summary>
    internal sealed class Qwen3AsrRuntime : IQwen3AsrRuntime
    {
        private readonly ILogger _logger;
        private readonly Qwen3AsrSessions _sessions;
        private readonly Qwen3AsrAudioEncoder _encoder;
        private readonly Qwen3AsrDecoder _decoder;
        private readonly Qwen3AsrTokenizer _tokenizer;
        private readonly Qwen3AsrMetadata _metadata;
        private readonly int _maxNewTokens;
        private int _disposed;

        public string? BuildStreamingPrefix(
            int chunkId,
            string accumulatedRaw,
            int unfixedChunkNum,
            int unfixedTokenNum)
            => StreamingPromptPlanner.BuildPrefix(
                chunkId,
                accumulatedRaw,
                unfixedChunkNum,
                unfixedTokenNum,
                _tokenizer);

        public Qwen3AsrRuntime(Qwen3ASROptions options, ILogger logger)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelPath);
            _logger = logger;
            _maxNewTokens = options.MaxNewTokens;

            string metadataPath = Path.Combine(options.ModelPath, "metadata.json");
            if (!File.Exists(metadataPath))
                throw new FileNotFoundException("metadata.json not found in the model directory (Qwen3ASROptions.ModelPath).", metadataPath);
            _metadata = JsonSerializer.Deserialize<Qwen3AsrMetadata>(File.ReadAllText(metadataPath))
                        ?? throw new InvalidOperationException("Failed to parse metadata.json.");

            string preprocPath = Path.Combine(options.ModelPath, "preprocessor_config.json");
            if (!File.Exists(preprocPath))
                throw new FileNotFoundException("preprocessor_config.json not found in the model directory (Qwen3ASROptions.ModelPath).", preprocPath);
            var preproc = JsonSerializer.Deserialize<PreprocessorConfig>(File.ReadAllText(preprocPath))
                          ?? throw new InvalidOperationException("Failed to parse preprocessor_config.json.");

            _sessions = new Qwen3AsrSessions(options.ModelPath, _metadata, purpose => BuildSessionOptions(options, purpose));
            try
            {
                _tokenizer = new Qwen3AsrTokenizer(options.ModelPath);
                var features = new WhisperFeatureExtractor(preproc);
                _encoder = new Qwen3AsrAudioEncoder(_metadata, _sessions, features);

                // GPU-resident KV cache via IOBinding is only beneficial when the model exports
                // its KV cache in *full* mode (output shape == input shape, the model rewrites
                // the cache in place each step). In delta mode the model outputs a tiny
                // [1, heads, 1, head_dim] slice per layer and we still need to scatter it into
                // the cache — which ORT C# has no GPU-side primitive for, so trying to keep
                // the cache on GPU would force a GPU→CPU readback of the *full* cache every
                // step. So: opt in only on CUDA + non-delta + EnableIoBinding.
                bool useGpuCache =
                    options.ExecutionProvider == OnnxExecutionProvider.Cuda
                    && options.EnableIoBinding
                    && !string.Equals(_metadata.DecodeCacheMode, "delta", StringComparison.OrdinalIgnoreCase);
                _decoder = new Qwen3AsrDecoder(_metadata, _sessions, useGpuCache, options.GpuDeviceId, _logger);
            }
            catch
            {
                // Construction failure must not leak the already-loaded ONNX graphs
                // (GBs of native/CUDA memory held by _sessions). Mirrors the
                // catch { Dispose(); throw; } pattern of Qwen3AsrSessions itself.
                // The decoder cleans up its own partial state in its ctor's catch.
                try { _encoder?.Dispose(); } catch { }
                _sessions.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Force-load the encoder/decoder graphs by running a single dummy pass over
        /// 1 s of low-level noise. Subsequent real calls then skip CUDA kernel JIT,
        /// constant upload and tensor-pool allocation — observed to remove the
        /// ~200 ms cold-start tax on the first encode and tens of ms on first decode.
        /// Failures are logged as warnings and swallowed; warmup is best-effort.
        /// </summary>
        public void Warmup(double audioSeconds = 1.0, int maxNewTokens = 4, CancellationToken ct = default)
        {
            EnsureNotDisposed();
            int sr = 16000;
            int len = Math.Max(sr / 4, (int)Math.Round(audioSeconds * sr));
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                // Low-amplitude noise (not pure silence) so log-mel doesn't degenerate
                // to a single-value spectrogram and skip realistic kernel paths.
                var waveform = new float[len];
                var rng = new Random(0xC0FFEE);
                for (int i = 0; i < len; i++)
                    waveform[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 1e-3);

                AudioFeatures audio = _encoder.Encode(waveform);
                double encodeMs = sw.Elapsed.TotalMilliseconds;
                if (audio.Rows == 0)
                {
                    _logger.LogInformation("Qwen3-ASR warmup 跳过: encoder 输出为空 ({ElapsedMs:F0}ms)", encodeMs);
                    return;
                }

                int[] inputIds = _tokenizer.BuildPromptTokens(
                    _metadata.AudioToken, audio.Rows, string.Empty, language: null, prefixText: null);
                int cap = Math.Max(1, Math.Min(maxNewTokens, _metadata.StaticCacheLen - inputIds.Length));
                _decoder.Decode(inputIds, audio, cap, ct);

                _logger.LogInformation(
                    "Qwen3-ASR warmup 完成: total={TotalMs:F0}ms (encode={EncodeMs:F0}ms, decode={DecodeMs:F0}ms, audio={AudioSec:F1}s prompt={PromptTok}tok)",
                    sw.Elapsed.TotalMilliseconds,
                    encodeMs,
                    sw.Elapsed.TotalMilliseconds - encodeMs,
                    audioSeconds,
                    inputIds.Length);
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Qwen3-ASR warmup 失败 ({ElapsedMs:F0}ms), 首次实际推理将承担冷启开销。", sw.Elapsed.TotalMilliseconds);
            }
        }

        /// <summary>
        /// Transcribe a mono 16 kHz waveform. The runtime is stateless across calls —
        /// concurrency must be serialised externally (via the recognition gate in the
        /// service). <paramref name="melState"/>, when supplied, lets callers amortise
        /// the log-mel feature extraction across the rolling re-transcribe loop of one
        /// streaming session; pass <c>null</c> for a one-shot encode.
        /// </summary>
        public Qwen3AsrTranscription Transcribe(
            ReadOnlySpan<float> waveform, string context, string? language, string? prefixText,
            CancellationToken ct,
            IncrementalMelState? melState = null)
        {
            EnsureNotDisposed();
            if (waveform.IsEmpty)
                return new Qwen3AsrTranscription(string.Empty, language ?? string.Empty,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0);

            var totalSw = System.Diagnostics.Stopwatch.StartNew();
            ct.ThrowIfCancellationRequested();
            AudioFeatures audio = _encoder.Encode(waveform, melState);
            if (audio.Rows == 0)
                return new Qwen3AsrTranscription(string.Empty, language ?? string.Empty,
                    audio.AudioSeconds, audio.EncodeSeconds, audio.MelSeconds,
                    0.0, 0.0, totalSw.Elapsed.TotalSeconds, 0, 0);

            int[] inputIds = _tokenizer.BuildPromptTokens(
                _metadata.AudioToken, audio.Rows, context, language, prefixText);

            ct.ThrowIfCancellationRequested();
            DecodeResult decode = _decoder.Decode(inputIds, audio, _maxNewTokens, ct);

            string raw = _tokenizer.Decode(decode.Tokens);
            ParseAsrOutput(raw, language, out string detectedLang, out string text);

            double totalSeconds = totalSw.Elapsed.TotalSeconds;
            double rtf = audio.AudioSeconds > 0 ? totalSeconds / audio.AudioSeconds : 0.0;

            // Guard the 9-arg structured log: when Information is filtered (typical for
            // benchmark/production), this avoids the params object?[] allocation plus
            // 9 box-of-double/int operations on every Transcribe call.
            if (_logger.IsEnabled(LogLevel.Information))
                _logger.LogInformation(
                    "Qwen3-ASR 单次识别完成: audio={AudioSec:F2}s total={TotalMs:F0}ms (encode={EncodeMs:F0}ms [mel={MelMs:F0}ms], prefill/TTFT={PrefillMs:F0}ms, decode={DecodeMs:F0}ms) RTF={Rtf:F2} prompt={Prompt}tok out={OutTokens}tok",
                    audio.AudioSeconds,
                    totalSeconds * 1000.0,
                    audio.EncodeSeconds * 1000.0,
                    audio.MelSeconds * 1000.0,
                    decode.PrefillSeconds * 1000.0,
                    decode.DecodeStepSeconds * 1000.0,
                    rtf,
                    inputIds.Length,
                    decode.Tokens.Length);

            if (_logger.IsEnabled(LogLevel.Debug))
                _logger.LogDebug(
                    "Qwen3-ASR 单次识别原文: rows={Rows} raw=\"{Raw}\"",
                    audio.Rows, raw);

            return new Qwen3AsrTranscription(
                Text: text,
                Language: detectedLang,
                AudioSeconds: audio.AudioSeconds,
                EncodeSeconds: audio.EncodeSeconds,
                MelSeconds: audio.MelSeconds,
                PrefillSeconds: decode.PrefillSeconds,
                DecodeStepSeconds: decode.DecodeStepSeconds,
                InferenceSeconds: totalSeconds,
                PromptTokens: inputIds.Length,
                TokenCount: decode.Tokens.Length);
        }

        /// <summary>
        /// Decode the raw chat-template output into (language, text). Mirrors the
        /// Python <c>parse_asr_output</c> helper.
        /// </summary>
        public static void ParseAsrOutput(string raw, string? userLanguage, out string language, out string text)
        {
            language = string.Empty;
            text = string.Empty;
            if (string.IsNullOrEmpty(raw)) return;
            string trimmed = raw.Trim();
            if (trimmed.Length == 0) return;

            if (!string.IsNullOrEmpty(userLanguage))
            {
                language = userLanguage!;
                text = trimmed;
                return;
            }

            const string tag = "<asr_text>";
            int idx = trimmed.IndexOf(tag, StringComparison.Ordinal);
            if (idx < 0) { text = trimmed; return; }

            string meta = trimmed.Substring(0, idx);
            string body = trimmed.Substring(idx + tag.Length).Trim();
            if (meta.ToLowerInvariant().Contains("language none"))
            {
                text = body;
                return;
            }

            foreach (var line in meta.Split('\n', StringSplitOptions.None))
            {
                string l = line.Trim();
                if (l.StartsWith("language ", StringComparison.OrdinalIgnoreCase))
                {
                    language = l.Substring("language ".Length).Trim();
                    break;
                }
            }
            text = body;
        }

        private void EnsureNotDisposed()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            _decoder.Dispose();
            _encoder.Dispose();
            _sessions.Dispose();
        }

        // Tracks the result of attempting to register a shared CUDA allocator at
        // OrtEnv level for each device id. Value true = successfully registered (caller
        // may set session.use_env_allocators=1); false = tried and failed, do not retry
        // and do NOT enable use_env_allocators on the SessionOptions (otherwise session
        // construction will throw "Failed to find allocator for device …").
        private static readonly object _sharedAllocatorLock = new();
        private static readonly System.Collections.Generic.Dictionary<int, bool> _sharedAllocatorState = new();

        /// <summary>
        /// Per-session <see cref="SessionOptions"/> factory. Encoder / conv_embed /
        /// audio_stack / prefill are dynamic-shape; the autoregressive decode loop has
        /// fully static shapes. On CUDA we additionally tighten the decode session's
        /// arena strategy to <c>kSameAsRequested</c> (kNextPowerOfTwo would double the
        /// allocation envelope on the very first decode step and waste GPU memory for
        /// shapes that never grow). Honors <see cref="OnnxExecutionOptions.SessionOptionsFactory"/>
        /// as a verbatim override when the caller has supplied one.
        /// </summary>
        private static SessionOptions BuildSessionOptions(Qwen3ASROptions options, SessionPurpose purpose)
        {
            if (options.SessionOptionsFactory is not null)
                return options.SessionOptionsFactory();

            if (options.ExecutionProvider != OnnxExecutionProvider.Cuda)
                return OnnxSessionFactory.Create(options, purpose);

            // CUDA: clone the tuning so the per-purpose override of arena_extend_strategy /
            // do_copy_in_default_stream does NOT leak back into the caller's CudaTuning
            // instance (or into other sessions sharing the same options object).
            var tuning = CloneCudaTuning(options.CudaTuning);
            if (purpose == SessionPurpose.StaticShapes)
            {
                // Static-shape decode: tight arena (no growth needed) and put H2D on its
                // own CUDA stream so the next-step KV-cache copy starts while the current
                // step's compute is still in flight. The "data visibility" caveat in the
                // CudaTuning XML applies to CUDA <= 11.0 with very old drivers; ORT 1.26
                // + CUDA 12 + modern drivers handles stream synchronisation correctly via
                // events, and ORT's own decoder samples ship with this flag off.
                tuning.ArenaExtendStrategy = "kSameAsRequested";
                tuning.DoCopyInDefaultStream = false;
            }

            // Register a process-shared CUDA allocator at OrtEnv level (idempotent) so
            // all three sessions (encoder, prefill, decode) share one BFCArena instead
            // of allocating three. Strategy = kNextPowerOfTwo for the shared arena —
            // encoder activations are large and variable, and a per-power-of-two ladder
            // keeps fragmentation low when the same buffer is recycled across encoder /
            // prefill / decode runs (the recognition gate serialises these calls).
            // Failure here is non-fatal — sessions still work, just without sharing.
            bool sharedRegistered = TryEnsureSharedCudaAllocator(options.GpuDeviceId, "kNextPowerOfTwo");

            var so = OnnxSessionFactory.CreateCuda(options.GpuDeviceId, options.ThreadsNum, tuning, purpose);
            if (sharedRegistered)
                so.AddSessionConfigEntry("session.use_env_allocators", "1");
            return so;
        }

        /// <summary>
        /// Register a CUDA allocator at <see cref="OrtEnv"/> level so multiple Qwen3-ASR
        /// sessions share a single BFCArena. Returns true if registration succeeded (or
        /// was already done in a prior call); false if the call threw — in which case the
        /// caller should NOT set <c>session.use_env_allocators</c>, since the EP would
        /// then bind to a non-existent env allocator and fail at session construction.
        /// </summary>
        private static bool TryEnsureSharedCudaAllocator(int deviceId, string arenaExtendStrategy)
        {
            lock (_sharedAllocatorLock)
            {
                if (_sharedAllocatorState.TryGetValue(deviceId, out bool cached)) return cached;

                OrtMemoryInfo? memInfo = null;
                OrtArenaCfg? arenaCfg = null;
                try
                {
                    memInfo = new OrtMemoryInfo(
                        OrtMemoryInfo.allocatorCUDA,
                        OrtAllocatorType.ArenaAllocator,
                        deviceId,
                        OrtMemType.Default);
                    // (maxMemory=0 → unlimited, extendStrategy=-1/initChunk=-1/maxDead=-1 → defaults;
                    //  the actual strategy is overridden by provider_options below.)
                    arenaCfg = new OrtArenaCfg(0, -1, -1, -1);
                    var providerOptions = new System.Collections.Generic.Dictionary<string, string>
                    {
                        ["device_id"] = deviceId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        ["arena_extend_strategy"] = arenaExtendStrategy,
                    };
                    OrtEnv.Instance().CreateAndRegisterAllocator(
                        "CUDAExecutionProvider", memInfo, arenaCfg, providerOptions);
                    _sharedAllocatorState[deviceId] = true;
                    return true;
                }
                catch
                {
                    // Older ORT builds, missing CUDA EP, or device unavailable. Cache the
                    // failure so we don't pay the exception cost on every subsequent
                    // BuildSessionOptions call, and so callers can skip use_env_allocators.
                    _sharedAllocatorState[deviceId] = false;
                    return false;
                }
                finally
                {
                    arenaCfg?.Dispose();
                    memInfo?.Dispose();
                }
            }
        }

        private static CudaExecutionTuning CloneCudaTuning(CudaExecutionTuning? source)
        {
            if (source is null) return new CudaExecutionTuning();
            return new CudaExecutionTuning
            {
                CudnnConvAlgoSearch = source.CudnnConvAlgoSearch,
                ArenaExtendStrategy = source.ArenaExtendStrategy,
                CudnnConvUseMaxWorkspace = source.CudnnConvUseMaxWorkspace,
                DoCopyInDefaultStream = source.DoCopyInDefaultStream,
                GpuMemLimitBytes = source.GpuMemLimitBytes,
                AdditionalOptions = source.AdditionalOptions is null
                    ? null
                    : new System.Collections.Generic.Dictionary<string, string>(source.AdditionalOptions),
            };
        }
    }

    /// <summary>
    /// One transcription pass result, including the parsed language tag and a full timing
    /// breakdown. <c>InferenceSeconds</c> is the total wall-clock time of <see cref="Qwen3AsrRuntime.Transcribe"/>;
    /// <c>PrefillSeconds</c> is the time-to-first-token (TTFT) measured from the start of the
    /// decoder prefill call.
    /// </summary>
    internal readonly record struct Qwen3AsrTranscription(
        string Text,
        string Language,
        double AudioSeconds,
        double EncodeSeconds,
        double MelSeconds,
        double PrefillSeconds,
        double DecodeStepSeconds,
        double InferenceSeconds,
        int PromptTokens,
        int TokenCount);
}
