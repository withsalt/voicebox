using System.Buffers;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;
using VoiceBox.Wakeup.OpenWakeWord.Internal;
using VoiceBox.Wakeup.OpenWakeWord.Models;

namespace VoiceBox.Wakeup.OpenWakeWord
{
    /// <summary>
    /// 基于 openWakeWord (https://github.com/dscripka/openWakeWord) 的流式唤醒词检测服务。
    /// 支持加载任意多个 ONNX 唤醒词模型, 共享同一份 mel/embedding 特征提取管线。
    /// 仅接收 16kHz 单声道 float32 PCM ([-1, 1] 振幅)。
    /// </summary>
    public sealed class OpenWakeWordService : IWakeupService
    {
        // 16kHz 下 80ms 一帧
        private const int ChunkSamples = 1280;
        private const float Int16ScaleFactor = 32768f;

        public event EventHandler<WakeupEventArgs>? OnWakeup;

        private readonly OpenWakeWordOptions _options;
        private readonly ILogger<OpenWakeWordService> _logger;

        private readonly AudioFeatureExtractor _featureExtractor;
        private readonly WakeWordDetector[] _detectors;
        private readonly float[] _scratchFeatureBuffer;

        private readonly ArrayPool<float> _arrayPool = ArrayPool<float>.Shared;

        // 锁住整个推理状态; 事件回调统一在锁外触发, 避免长耗时 handler 卡住音频管线。
        private readonly object _lock = new();
        // volatile: FeedAudio 快速路径在锁外读取本标志, 需保证 Stop() 在其它线程写入的 false 能及时
        // 可见 (与 SherpaOnnxKwsService 保持一致)。锁内仍会复检, 故此处仅作跨线程可见性保证。
        private volatile bool _isListening;
        private bool _disposed;
        private long _processedSampleIndex;

        // 在本次 Process 内累积的待派发事件; 出锁后再触发。
        private readonly List<WakeupEventArgs> _pendingEvents = new();

        public bool IsListening
        {
            get { lock (_lock) return _isListening; }
        }

        public OpenWakeWordService(IOptions<OpenWakeWordOptions> options)
            : this(options, NullLogger<OpenWakeWordService>.Instance)
        {
        }

        public OpenWakeWordService(IOptions<OpenWakeWordOptions> options, ILogger<OpenWakeWordService> logger)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<OpenWakeWordService>.Instance;

            if (_options.SampleRate != 16000)
                throw new ArgumentException("OpenWakeWord only supports 16000Hz sample rate.", nameof(options));
            if (_options.ChannelCount != 1)
                throw new ArgumentException("OpenWakeWord only supports mono audio.", nameof(options));
            if (_options.WakeWords == null || _options.WakeWords.Count == 0)
                throw new ArgumentException("At least one wake word must be configured.", nameof(options));

            _featureExtractor = new AudioFeatureExtractor(
                _options.MelSpectrogramModelPath,
                _options.EmbeddingModelPath,
                _options);

            _detectors = _options.WakeWords
                .Select(def => new WakeWordDetector(def, _options.SampleRate, _options))
                .ToArray();

            int maxFeatureFrames = _detectors.Max(d => d.FeatureFrames);
            _scratchFeatureBuffer = new float[maxFeatureFrames * 96];

            _logger.LogInformation(
                "OpenWakeWord initialized with {Count} wakeword(s): {Keywords}",
                _detectors.Length,
                string.Join(", ", _detectors.Select(d => d.Definition.Keyword)));
        }

        public void Start()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_isListening) return;
                _featureExtractor.Reset();
                foreach (var d in _detectors) d.Reset();
                _processedSampleIndex = 0;
                _isListening = true;
            }
            _logger.LogDebug("OpenWakeWord listening started");
        }

        public void Stop()
        {
            lock (_lock)
            {
                if (!_isListening) return;
                _isListening = false;
            }
            _logger.LogDebug("OpenWakeWord listening stopped");
        }

        public void SetThreshold(float threshold)
        {
            ValidateThreshold(threshold);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                foreach (var detector in _detectors)
                    detector.Definition.Threshold = threshold;
            }

            _logger.LogInformation("OpenWakeWord threshold updated to {Threshold} for {Count} wakeword(s)",
                threshold, _detectors.Length);
        }

        public void FeedAudio(ReadOnlySpan<float> audioData)
        {
            if (audioData.IsEmpty) return;

            // 快速路径: 不在监听状态时直接返回, 避免无谓的内存/扫描开销。
            // 注意此处的 IsListening 是一次性快照, 若与 Stop/Start 竞争则最差代价只是一次空跑。
            if (!_isListening) return;

            float[] rented = _arrayPool.Rent(audioData.Length);
            List<WakeupEventArgs>? pendingSnapshot = null;
            try
            {
                var scaled = rented.AsSpan(0, audioData.Length);
                for (int i = 0; i < audioData.Length; i++)
                {
                    float v = audioData[i] * Int16ScaleFactor;
                    if (v > 32767f) v = 32767f;
                    else if (v < -32768f) v = -32768f;
                    scaled[i] = v;
                }

                pendingSnapshot = ProcessScaledAudio(scaled);
            }
            finally
            {
                _arrayPool.Return(rented);
            }

            if (pendingSnapshot != null && pendingSnapshot.Count > 0)
            {
                DispatchEvents(pendingSnapshot);
            }
        }

        private List<WakeupEventArgs>? ProcessScaledAudio(ReadOnlySpan<float> scaled)
        {
            lock (_lock)
            {
                if (_disposed || !_isListening) return null;

                _pendingEvents.Clear();

                int offset = 0;
                while (offset < scaled.Length)
                {
                    // 每次最多塞一个 chunk (1280) 的量, 让每帧推理能在它对应的 _processedSampleIndex 上完成。
                    int take = Math.Min(ChunkSamples, scaled.Length - offset);
                    int newFrames = _featureExtractor.Process(scaled.Slice(offset, take));
                    offset += take;
                    _processedSampleIndex += take;

                    if (newFrames > 0)
                    {
                        RunDetectorsForNewFrames();
                    }
                }

                if (_pendingEvents.Count == 0) return null;
                var copy = new List<WakeupEventArgs>(_pendingEvents);
                _pendingEvents.Clear();
                return copy;
            }
        }

        private void RunDetectorsForNewFrames()
        {
            foreach (var detector in _detectors)
            {
                int n = detector.FeatureFrames;
                if (_featureExtractor.EmbeddingFrames < n) continue;

                Span<float> feats = _scratchFeatureBuffer.AsSpan(0, n * 96);
                if (!_featureExtractor.TryCopyLatestFeatures(n, feats))
                    continue;

                try
                {
                    if (detector.Predict(feats, _processedSampleIndex, out float score))
                    {
                        QueueWakeupEvent(detector.Definition, score);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Wakeword inference failed for '{Keyword}'.", detector.Definition.Keyword);
                }
            }
        }

        private void QueueWakeupEvent(WakeWordModelDefinition def, float score)
        {
            int scorePercent = (int)Math.Clamp(MathF.Round(score * 100f), 0f, 100f);
            int thresholdPercent = (int)Math.Clamp(MathF.Round(def.Threshold * 100f), 0f, 100f);
            long startTimeMs = Math.Max(0, _processedSampleIndex * 1000L / _options.SampleRate);

            _pendingEvents.Add(new WakeupEventArgs(
                keyword: def.Keyword,
                score: scorePercent,
                threshold: thresholdPercent,
                durationMs: 80,
                startTimeMs: startTimeMs));

            _logger.LogInformation(
                "Wakeword hit: keyword={Keyword} score={Score} threshold={Threshold} t={StartTimeMs}ms",
                def.Keyword, scorePercent, thresholdPercent, startTimeMs);
        }

        private void DispatchEvents(List<WakeupEventArgs> events)
        {
            var handler = OnWakeup;
            if (handler == null) return;

            foreach (var args in events)
            {
                try
                {
                    handler.Invoke(this, args);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "OnWakeup handler exception for '{Keyword}'", args.Keyword);
                }
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _isListening = false;
                foreach (var d in _detectors) d.Dispose();
                _featureExtractor.Dispose();
            }
        }

        private static void ValidateThreshold(float threshold)
        {
            if (!float.IsFinite(threshold) || threshold <= 0f || threshold > 1f)
                throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "Threshold must be in (0, 1].");
        }
    }
}
