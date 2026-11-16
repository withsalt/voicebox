using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.ASR.FireRedASR2.Internal;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.ASR.FireRedASR2.Pump;
using VoiceBox.ASR.FireRedASR2.Session;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.FireRedASR2
{
    /// <summary>
    /// Offline FireRed ASR service. Owns the loaded recognizer (a single ONNX model shared
    /// across sessions), while the base service manages session lifetimes. Per-session work
    /// happens on the pump and a final synchronous inference pass at <see cref="FinishSessionAsync"/>.
    /// </summary>
    public sealed class FireRed2ASRService : BaseASRService
    {
        private readonly FireRed2ASROptions _options;
        private readonly ILogger<FireRed2ASRService> _logger;
        private readonly OfflineRecognizer _recognizer;
        private readonly SemaphoreSlim _recognitionGate;

        public FireRed2ASRService(IOptions<FireRed2ASROptions> options, ILogger<FireRed2ASRService>? logger = null)
            : this(options, new FileSystemOfflineRecognizerFactory(), logger)
        {
        }

        internal FireRed2ASRService(
            IOptions<FireRed2ASROptions> options,
            IOfflineRecognizerFactory recognizerFactory,
            ILogger<FireRed2ASRService>? logger = null)
            : base(GetMaxConcurrentSessions(options))
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(recognizerFactory);

            _options = options.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<FireRed2ASRService>.Instance;

            ValidateOptions(_options);

            _recognizer = recognizerFactory.Create(_options);
            _recognitionGate = new SemaphoreSlim(_options.MaxConcurrentRecognitions, _options.MaxConcurrentRecognitions);
        }

        protected override AsrSession CreateSession(string sessionId)
        {
            OfflineStream stream = _recognizer.CreateOfflineStream();
            try
            {
                return new FireRedSession(sessionId, stream, _options.MaxBufferChunks,
                    _recognizer, _recognitionGate, _options.FinishDrainTimeoutSec, _logger);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }

        protected override void DisposeProvider()
        {
            try { _recognizer.Dispose(); }
            finally { _recognitionGate.Dispose(); }
        }

        private static void ValidateOptions(FireRed2ASROptions options)
        {
            if (options.MaxBufferChunks < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxBufferChunks must be ≥ 1.");
            if (options.MaxConcurrentSessions < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentSessions must be ≥ 1.");
            if (options.MaxConcurrentRecognitions < 1)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentRecognitions must be ≥ 1.");
        }

        private static int GetMaxConcurrentSessions(IOptions<FireRed2ASROptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);
            return options.Value?.MaxConcurrentSessions
                ?? throw new ArgumentException("FireRed2ASROptions cannot be null.", nameof(options));
        }
    }
}
