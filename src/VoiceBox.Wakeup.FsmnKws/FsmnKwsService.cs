using System.Buffers;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;
using VoiceBox.Wakeup.FsmnKws.Internal;
using VoiceBox.Wakeup.FsmnKws.Models;

namespace VoiceBox.Wakeup.FsmnKws
{
    /// <summary>
    /// 基于 FunASR FSMN+CTC 字建模 (小云模型) 的流式唤醒词服务。
    /// 单实例 + 单流式: 同一时刻只处理一路 16kHz/mono/float32 ([-1, 1]) 音频。
    /// <para>
    /// 流程: FeedAudio → kaldi fbank(80, hamming) → LFR(5,3)+CMVN → 流式 FSMN ONNX (带 cache,
    /// 每输出帧 30ms) → CTC prefix beam search 关键词检测 → OnWakeup 事件。
    /// 唤醒词可任意配置 (中文 char 全集建模), 多关键词共享一次解码。
    /// </para>
    /// </summary>
    public sealed class FsmnKwsService : IWakeupService
    {
        /// <summary>每个模型输出帧对应的毫秒数 (LFR 后帧移 30ms)。</summary>
        private const int OutputFrameMs = 30;

        public event EventHandler<WakeupEventArgs>? OnWakeup;

        private readonly FsmnKwsOptions _options;
        private readonly ILogger<FsmnKwsService> _logger;

        private readonly StreamingKaldiFbank _fbank = new();
        private readonly LfrCmvnFrontend _frontend;
        private readonly StreamingFsmnModel _model;
        private readonly CtcKeywordDecoder _decoder;

        private long _lastKeywordTimeMs;
        private string _lastKeyword = string.Empty;

        private readonly ArrayPool<float> _arrayPool = ArrayPool<float>.Shared;
        private readonly object _lock = new();
        private volatile bool _isListening;
        private bool _disposed;
        private readonly List<WakeupEventArgs> _pendingEvents = new();

        public bool IsListening { get { lock (_lock) return _isListening; } }

        public FsmnKwsService(IOptions<FsmnKwsOptions> options)
            : this(options, NullLogger<FsmnKwsService>.Instance)
        {
        }

        public FsmnKwsService(IOptions<FsmnKwsOptions> options, ILogger<FsmnKwsService> logger)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<FsmnKwsService>.Instance;

            if (_options.SampleRate != 16000)
                throw new ArgumentException("FsmnKws only supports 16000Hz sample rate.", nameof(options));
            if (_options.ChannelCount != 1)
                throw new ArgumentException("FsmnKws only supports mono audio.", nameof(options));
            if (_options.SampleFormat != SampleFormat.Float32)
                throw new ArgumentException("FsmnKws only supports Float32 audio samples.", nameof(options));
            if (string.IsNullOrWhiteSpace(_options.ModelPath))
                throw new ArgumentException("FsmnKws requires ModelPath.", nameof(options));
            if (!float.IsFinite(_options.Threshold) || _options.Threshold <= 0f || _options.Threshold > 1f)
                throw new ArgumentException("Threshold must be in (0, 1].", nameof(options));
            if (_options.ScoreBeam < 1 || _options.PathBeam < 1)
                throw new ArgumentException("ScoreBeam/PathBeam must be >= 1.", nameof(options));
            if (_options.ScoreBeam > StreamingFsmnModel.OutputDim)
                throw new ArgumentException($"ScoreBeam must be <= vocabulary size ({StreamingFsmnModel.OutputDim}).", nameof(options));
            if (!float.IsFinite(_options.TokenProbFloor) || _options.TokenProbFloor < 0f || _options.TokenProbFloor > 1f)
                throw new ArgumentException("TokenProbFloor must be in [0, 1].", nameof(options));
            if (!float.IsFinite(_options.MaxKeywordSeconds) || _options.MaxKeywordSeconds <= 0f)
                throw new ArgumentException("MaxKeywordSeconds must be a positive finite value.", nameof(options));
            double maxSpanFrameValue = _options.MaxKeywordSeconds * 1000d / OutputFrameMs;
            if (maxSpanFrameValue > int.MaxValue)
                throw new ArgumentException("MaxKeywordSeconds is too large.", nameof(options));
            if (_options.DebounceMs < 0)
                throw new ArgumentException("DebounceMs must be >= 0.", nameof(options));

            // 唤醒词从模型目录的 kws_meta.json 读取 (不再由调用方配置); 缺文件/缺字段直接抛异常。
            string metaPath = !string.IsNullOrWhiteSpace(_options.MetaFilePath)
                ? _options.MetaFilePath
                : FsmnKwsMetadata.ResolveMetaPath(_options.ModelPath);
            var meta = FsmnKwsMetadata.Load(metaPath);

            if (meta.OovChars.Count > 0)
                _logger.LogWarning(
                    "kws_meta.json 唤醒词 '{Keyword}' 含 OOV 字 [{Oov}] (不在模型 token 表), 这些字无法被唤醒。",
                    meta.Keyword, string.Join(", ", meta.OovChars));

            int[] ids = meta.KeywordTokenIds.ToArray();
            foreach (int id in ids)
            {
                if (id < 0 || id >= StreamingFsmnModel.OutputDim)
                    throw new InvalidDataException(
                        $"kws_meta.json keyword token id {id} out of range [0, {StreamingFsmnModel.OutputDim}): {metaPath}");
            }

            var entries = new[]
            {
                new CtcKeywordDecoder.KeywordEntry(meta.Keyword, ids, _options.Threshold),
            };

            _frontend = new LfrCmvnFrontend(meta.CmvnAddShift, meta.CmvnRescale);
            _model = new StreamingFsmnModel(_options);

            long maxSpanFrames = Math.Max(1, (long)maxSpanFrameValue);
            _decoder = new CtcKeywordDecoder(
                entries, _options.ScoreBeam, _options.PathBeam, _options.TokenProbFloor, maxSpanFrames);

            _logger.LogInformation(
                "FsmnKws initialized: keyword='{Keyword}' (tokens=[{Ids}]), threshold={Threshold}, " +
                "scoreBeam={ScoreBeam}, pathBeam={PathBeam}, maxSpan={MaxSpan}f",
                meta.Keyword, string.Join(", ", ids),
                _options.Threshold, _options.ScoreBeam, _options.PathBeam, maxSpanFrames);
        }

        public void Start()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_isListening) return;
                ResetInternal();
                _isListening = true;
            }
            _logger.LogDebug("FsmnKws listening started");
        }

        public void Stop()
        {
            lock (_lock)
            {
                if (!_isListening) return;
                _isListening = false;
            }
            _logger.LogDebug("FsmnKws listening stopped");
        }

        public void SetThreshold(float threshold)
        {
            ValidateThreshold(threshold);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _decoder.SetThreshold(threshold);
            }

            _logger.LogInformation("FsmnKws threshold updated to {Threshold}", threshold);
        }

        public void FeedAudio(ReadOnlySpan<float> audioData)
        {
            if (audioData.IsEmpty) return;
            if (!_isListening) return; // 快速路径

            float[] rented = _arrayPool.Rent(audioData.Length);
            List<WakeupEventArgs>? pending = null;
            try
            {
                audioData.CopyTo(rented.AsSpan(0, audioData.Length));
                pending = ProcessAudio(rented.AsSpan(0, audioData.Length));
            }
            finally
            {
                _arrayPool.Return(rented);
            }

            if (pending != null && pending.Count > 0) DispatchEvents(pending);
        }

        private List<WakeupEventArgs>? ProcessAudio(ReadOnlySpan<float> audio)
        {
            lock (_lock)
            {
                if (_disposed || !_isListening) return null;
                _pendingEvents.Clear();

                // 1) fbank + LFR/CMVN: 直接写入模型输入缓冲 (零中转拷贝)
                _fbank.AcceptWaveform(audio);
                int frameCount = _frontend.PendingFrameCount(_fbank);
                if (frameCount == 0) return null;

                Span<float> feats = _model.GetFeatsBuffer(frameCount);
                int produced = _frontend.ProduceFrames(_fbank, feats);

                // 2) 一次流式推理 + 逐输出帧解码
                ReadOnlySpan<float> probs = _model.Run(produced);
                for (int i = 0; i < produced; i++)
                {
                    var hit = _decoder.Step(probs.Slice(i * StreamingFsmnModel.OutputDim, StreamingFsmnModel.OutputDim));
                    if (hit is { } d)
                    {
                        QueueKeywordEvent(d);
                        _decoder.Reset(); // 与 funasr 行为一致: 命中后重置搜索, 保留声学 cache
                    }
                }

                if (_pendingEvents.Count == 0) return null;
                var snapshot = new List<WakeupEventArgs>(_pendingEvents);
                _pendingEvents.Clear();
                return snapshot;
            }
        }

        private void QueueKeywordEvent(in CtcKeywordDecoder.Detection d)
        {
            // 输出帧 j 对应输入音频时间 (j - OutputDelayFrames) * 30ms
            long startMs = Math.Max(0, d.StartFrame - StreamingFsmnModel.OutputDelayFrames) * OutputFrameMs;
            long endMs = Math.Max(0, d.EndFrame - StreamingFsmnModel.OutputDelayFrames + 1) * OutputFrameMs;

            if (_lastKeyword == d.Keyword && endMs - _lastKeywordTimeMs < _options.DebounceMs)
            {
                _logger.LogDebug("Debounced keyword '{Keyword}' at t={Ms}ms", d.Keyword, startMs);
                return;
            }
            _lastKeyword = d.Keyword;
            _lastKeywordTimeMs = endMs;

            int scorePercent = (int)Math.Clamp(MathF.Round(d.Score * 100f), 0f, 100f);
            int thresholdPercent = (int)Math.Clamp(MathF.Round(d.Threshold * 100f), 0f, 100f);

            _pendingEvents.Add(new WakeupEventArgs(
                keyword: d.Keyword,
                score: scorePercent,
                threshold: thresholdPercent,
                durationMs: Math.Max(0, endMs - startMs),
                startTimeMs: startMs));

            _logger.LogInformation(
                "Wakeword hit: keyword={Keyword} score={Score}% threshold={Threshold}% t={StartTimeMs}ms",
                d.Keyword, scorePercent, thresholdPercent, startMs);
        }

        private void ResetInternal()
        {
            _fbank.Reset();
            _frontend.Reset();
            _model.ResetCache();
            _decoder.ResetAll();
            _lastKeyword = string.Empty;
            _lastKeywordTimeMs = 0;
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
                    _logger.LogWarning(ex, "OnWakeup handler exception for keyword '{Keyword}'", args.Keyword);
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
                _model.Dispose();
            }
        }

        private static void ValidateThreshold(float threshold)
        {
            if (!float.IsFinite(threshold) || threshold <= 0f || threshold > 1f)
                throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "Threshold must be in (0, 1].");
        }
    }
}
