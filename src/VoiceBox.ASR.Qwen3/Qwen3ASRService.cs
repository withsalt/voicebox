using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.ASR.Qwen3.Internal;
using VoiceBox.ASR.Qwen3.Pump;
using VoiceBox.ASR.Qwen3.Session;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Qwen3
{
    /// <summary>
    /// Production-grade streaming ASR service for Qwen3-ASR ONNX exports.
    /// </summary>
    /// <remarks>
    /// Architecture
    /// ------------
    /// The service is composed of three collaborating layers:
    ///
    ///   * <see cref="Qwen3AsrRuntime"/> — owns the heavy ONNX sessions (encoder
    ///     + autoregressive decoder + KV cache), tokenizer and Whisper feature
    ///     extractor. One instance per service; stateless across calls.
    ///   * <see cref="Qwen3AsrSession"/> — per-stream state, ingress audio channel,
    ///     and result channel. The shared service manager retains sessions until release.
    ///   * <see cref="Qwen3AsrAudioPump"/> — background loop draining audio,
    ///     accumulating the waveform and triggering rolling re-transcription.
    ///
    /// Streaming semantics
    /// -------------------
    /// Qwen3-ASR is an offline encoder/decoder model: there is no chunkwise
    /// causal decoding. To still expose <see cref="IStreamASRService"/> we adopt
    /// the same windowed re-transcription pattern as the upstream Python
    /// reference: every <c>ChunkSizeSec</c> of accumulated audio we re-encode
    /// from the start and re-decode, optionally seeding the prompt with the
    /// previously decoded text (minus its last few tokens) so the model keeps
    /// continuity without drifting near the bleeding edge.
    ///
    /// Concurrency
    /// -----------
    /// ONNX inference is heavy (especially on CPU), and the runtime owns mutable
    /// encoder/decoder scratch state. A service instance therefore accepts only
    /// one concurrent inference.
    /// </remarks>
    public sealed class Qwen3ASRService : BaseASRService, IStreamASRService
    {
        private readonly Qwen3ASROptions _options;
        private readonly ILogger<Qwen3ASRService> _logger;
        private readonly IQwen3AsrRuntime _runtime;
        private readonly SemaphoreSlim _recognitionGate;

        public Qwen3ASRService(
            IOptions<Qwen3ASROptions> options,
            ILogger<Qwen3ASRService>? logger = null)
            : base(options?.Value?.MaxConcurrentSessions ?? 8)
        {
            ArgumentNullException.ThrowIfNull(options);
            _options = options.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<Qwen3ASRService>.Instance;

            ValidateOptions(_options);
            _runtime = new Qwen3AsrRuntime(_options, _logger);
            _recognitionGate = new SemaphoreSlim(_options.MaxConcurrentInferences, _options.MaxConcurrentInferences);
            _runtime.Warmup();
        }

        internal Qwen3ASRService(
            Qwen3ASROptions options,
            IQwen3AsrRuntime runtime,
            ILogger<Qwen3ASRService>? logger = null)
            : base(options?.MaxConcurrentSessions ?? 8)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<Qwen3ASRService>.Instance;

            ValidateOptions(_options);
            _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
            _recognitionGate = new SemaphoreSlim(_options.MaxConcurrentInferences, _options.MaxConcurrentInferences);
            _runtime.Warmup();
        }

        protected override AsrSession CreateSession(string sessionId)
        {
            return new Qwen3AsrSession(sessionId, _options.DefaultContext,
                NormalizeLanguage(_options.ForceLanguage),
                Math.Max(1, (int)Math.Round(_options.ChunkSizeSec * 16000)),
                _options.UnfixedChunkNum, _options.UnfixedTokenNum,
                _options.MaxNewTokens, _options.MaxBufferChunks, _runtime,
                _recognitionGate, _options.FinishDrainTimeoutSec, _logger);
        }

        protected override void DisposeProvider()
        {
            try { _runtime.Dispose(); }
            finally { _recognitionGate.Dispose(); }
        }

        public async IAsyncEnumerable<AsrResult> GetStreamResultAsync(
            string sessionId,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
            EnsureServiceActive();

            if (!TryGetSession<Qwen3AsrSession>(sessionId, out var session))
                throw new InvalidOperationException($"Session not found: {sessionId}");

            await foreach (var item in session.ResultReader.ReadAllAsync(ct).ConfigureAwait(false))
                yield return item;
        }

        private static void ValidateOptions(Qwen3ASROptions options)
        {
            if (string.IsNullOrWhiteSpace(options.ModelPath))
                throw new ArgumentException("Qwen3ASROptions.ModelPath must be set.");
            if (options.MaxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxNewTokens must be ≥ 1.");
            if (options.MaxBufferChunks < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxBufferChunks must be ≥ 1.");
            if (options.MaxConcurrentSessions < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentSessions must be ≥ 1.");
            if (options.MaxConcurrentInferences != 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentInferences must be exactly 1 because a runtime instance cannot execute concurrent inference safely.");
            if (options.ChunkSizeSec <= 0)
                throw new ArgumentOutOfRangeException(nameof(options), "ChunkSizeSec must be > 0.");
            if (options.UnfixedChunkNum < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "UnfixedChunkNum must be ≥ 0.");
            if (options.UnfixedTokenNum < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "UnfixedTokenNum must be ≥ 0.");
        }

        private static string? NormalizeLanguage(string? language)
        {
            if (string.IsNullOrWhiteSpace(language)) return null;
            string s = language.Trim();
            return char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant();
        }

    }
}
