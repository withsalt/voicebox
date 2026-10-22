using System.Buffers;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.WakeupService.Models;
using VoiceBox.Wakeup.SherpaOnnxKws.Internal;
using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws
{
    /// <summary>
    /// 基于 sherpa-onnx KWS Zipformer-Transducer 的流式命令词唤醒服务。
    /// 单实例 + 单流式: 同一时刻只处理一路 16kHz/mono/float32 ([-1, 1]) 音频。
    /// <para>
    /// 流程: FeedAudio -> kaldi fbank 特征 -> zipformer encoder (chunk_size=45/shift=32) ->
    /// modified beam-search + Aho-Corasick context graph -> 命中关键词 -> 触发 OnWakeup 事件。
    /// </para>
    /// </summary>
    public sealed class SherpaOnnxKwsService : IWakeupService
    {
        public event EventHandler<WakeupEventArgs>? OnWakeup;

        private readonly SherpaOnnxKwsOptions _options;
        private readonly ILogger<SherpaOnnxKwsService> _logger;

        private readonly TransducerModel _model;
        private readonly SymbolTable _symbols;
        private readonly ContextGraph _contextGraph;
        private readonly TransducerKeywordDecoder _decoder;
        private readonly OnlineKaldiFbank _fbank;
        private readonly Dictionary<string, int> _keywordDebounceMs;
        private float _threshold;

        // 解码状态
        private EncoderState _encoderState;
        private KeywordDecoderResult _decoderResult;
        // 绝对帧号用 long: 每帧 10ms, 二者之和是绝对 fbank 帧索引, 只在 ResetInternal(Start) 清零,
        // int 约 248 天连续运行后会溢出转负, 破坏 CopyFrames/Pop 的帧号算术 (越界/不再丢帧)。
        private long _numProcessedFrames;
        private long _startFrameIndex; // 用于计算事件 StartTimeMs (帧->秒)
        private long _lastKeywordTimeMs;
        private string _lastKeyword = string.Empty;
        // 静音自动 Reset 计数 (单位: subsampled 帧). 与 sherpa-onnx 一致: trailing_silence > 1.5s 自动重置。
        private readonly int _silenceResetTrailingBlanks;

        private readonly ArrayPool<float> _arrayPool = ArrayPool<float>.Shared;
        private readonly object _lock = new();
        // 用 volatile 让 FeedAudio 快速路径 (无锁读取) 看到最近的 Start/Stop。
        // 即使发生竞争, 最差也只是漏处理一个 chunk; 慢路径会在锁内复核。
        private volatile bool _isListening;
        private bool _disposed;

        // 用于在锁外派发事件; ProcessAudio 内部累积, 退出锁后再触发
        private readonly List<WakeupEventArgs> _pendingEvents = new();

        public bool IsListening { get { lock (_lock) return _isListening; } }

        public SherpaOnnxKwsService(IOptions<SherpaOnnxKwsOptions> options)
            : this(options, NullLogger<SherpaOnnxKwsService>.Instance)
        {
        }

        public SherpaOnnxKwsService(IOptions<SherpaOnnxKwsOptions> options, ILogger<SherpaOnnxKwsService> logger)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<SherpaOnnxKwsService>.Instance;

            if (_options.SampleRate != 16000)
                throw new ArgumentException("SherpaOnnxKws only supports 16000Hz sample rate.", nameof(options));
            if (_options.ChannelCount != 1)
                throw new ArgumentException("SherpaOnnxKws only supports mono audio.", nameof(options));
            if (_options.MaxActivePaths < 1)
                throw new ArgumentException("MaxActivePaths must be >= 1.", nameof(options));
            if (_options.KeywordsThreshold <= 0f || _options.KeywordsThreshold > 1f)
                throw new ArgumentException("KeywordsThreshold must be in (0, 1].", nameof(options));

            _threshold = _options.KeywordsThreshold;

            _model = new TransducerModel(_options);
            _symbols = new SymbolTable(_options.TokensFilePath);

            // 解析 + 编码全部关键词
            var encoded = KeywordEncoder.EncodeAll(_options, _symbols);
            var tokenIdLists = new List<int[]>(encoded.Count);
            var scores = new List<float>(encoded.Count);
            var phrases = new List<string>(encoded.Count);
            var thresholds = new List<float>(encoded.Count);
            _keywordDebounceMs = new Dictionary<string, int>(encoded.Count, StringComparer.Ordinal);
            foreach (var ek in encoded)
            {
                tokenIdLists.Add(ek.TokenIds);
                scores.Add(ek.BoostScore);
                phrases.Add(ek.Phrase);
                thresholds.Add(ek.Threshold);
                _keywordDebounceMs[ek.Phrase] = ek.DebounceMs > 0 ? ek.DebounceMs : _options.DebounceMs;
            }
            _contextGraph = new ContextGraph(
                tokenIdLists, _options.KeywordsScore, _options.KeywordsThreshold,
                scores, phrases, thresholds);

            int unkId = _symbols.TryGetId("<unk>", out int u) ? u : -1;
            _decoder = new TransducerKeywordDecoder(
                _model, _options.MaxActivePaths, _options.NumTrailingBlanks, unkId,
                _options.EmitImmediately);

            _fbank = new OnlineKaldiFbank();
            _encoderState = _model.CreateInitialEncoderState();
            _decoderResult = _decoder.GetEmptyResult();
            // 把初始 hyp 的 context state 指到 root
            foreach (var h in _decoderResult.Hyps.Values) h.ContextState = _contextGraph.Root;

            // 1.5s / (10ms * subsampling_factor 4) = 37.5 个 subsampled 帧
            _silenceResetTrailingBlanks = Math.Max(1,
                (int)(_options.ResetAfterSilenceSeconds / (_model.SubsamplingFactor * 0.01f)));

            _logger.LogInformation(
                "SherpaOnnxKws initialized: keywords={Count} [{Names}], threshold={Threshold}, " +
                "beam={Beam}, num_trailing_blanks={Trailing}, chunk_size={Chunk}, chunk_shift={Shift}",
                encoded.Count, string.Join(", ", encoded.Select(e => e.Phrase)),
                _options.KeywordsThreshold, _options.MaxActivePaths, _options.NumTrailingBlanks,
                _model.ChunkSize, _model.ChunkShift);
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
            _logger.LogDebug("SherpaOnnxKws listening started");
        }

        public void Stop()
        {
            lock (_lock)
            {
                if (!_isListening) return;
                _isListening = false;
            }
            _logger.LogDebug("SherpaOnnxKws listening stopped");
        }

        public void SetThreshold(float threshold)
        {
            ValidateThreshold(threshold);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _threshold = threshold;
                _contextGraph.SetThreshold(threshold);
            }

            _logger.LogInformation("SherpaOnnxKws threshold updated to {Threshold}", threshold);
        }

        public void FeedAudio(ReadOnlySpan<float> audioData)
        {
            if (audioData.IsEmpty) return;
            if (!_isListening) return; // 快速路径

            // 拷贝出来, 避免在锁内长期持有外部的 Span
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

                // 1) 累积特征
                _fbank.AcceptWaveform(16000, audio);

                // 2) 循环处理足够 chunk 的特征
                while (true)
                {
                    // 已生成的绝对帧数 (fbank 内部从 0 起算的总帧数)
                    long framesReady = _fbank.NumFramesReady;
                    // 下一个 chunk 的起始绝对帧索引
                    long absFrameStart = _numProcessedFrames + _startFrameIndex;
                    if (absFrameStart + _model.ChunkSize > framesReady) break;

                    int chunkFrames = _model.ChunkSize;
                    int featDim = _model.FeatureDim;
                    float[] features = new float[chunkFrames * featDim];
                    _fbank.CopyFrames(absFrameStart, chunkFrames, features);

                    // 3) 跑一次 encoder
                    var (encOut, encShape) = _model.RunEncoder(features, _encoderState);
                    // encShape = [1, T', joinerDim]
                    _numProcessedFrames += _model.ChunkShift;

                    // 4) 跑解码
                    _decoder.Decode(encOut, encShape, _contextGraph, _decoderResult);

                    // 4.5) 诊断: 几乎命中但被阈值/尾部静音拦下的事件 (Debug 级别, 排查唤醒失败)
                    if (_decoderResult.NearMisses.Count > 0)
                    {
                        foreach (var nm in _decoderResult.NearMisses)
                        {
                            string heard = RenderHeardTokens(nm.HeardTokenIds);
                            _logger.LogDebug(
                                "Near-miss keyword='{Phrase}' heard='{Heard}' ysProb={YsProb:F3} threshold={Th:F3} " +
                                "trailingBlanks={Blanks}/{Required}",
                                nm.Phrase, heard, nm.YsProb, nm.AcThreshold,
                                nm.NumTrailingBlanks, nm.RequiredTrailingBlanks);
                        }
                        _decoderResult.NearMisses.Clear();
                    }

                    // 5) 是否命中?
                    if (!string.IsNullOrEmpty(_decoderResult.Keyword))
                    {
                        QueueKeywordEvent(_decoderResult);
                        // 重置解码器 (与 sherpa-onnx C++ 的 Reset 等价)
                        ResetDecoderOnly();
                    }
                    else
                    {
                        // 6) 长时间静音 (trailing blanks > 阈值) 自动重置, 与 sherpa-onnx 行为一致
                        if (_decoderResult.NumTrailingBlanks > _silenceResetTrailingBlanks)
                        {
                            ResetDecoderOnly();
                        }
                    }
                }

                // 释放过早的特征帧缓存 (减少内存占用)
                _fbank.Pop(_numProcessedFrames + _startFrameIndex);

                if (_pendingEvents.Count == 0) return null;
                var snapshot = new List<WakeupEventArgs>(_pendingEvents);
                _pendingEvents.Clear();
                return snapshot;
            }
        }

        private void QueueKeywordEvent(KeywordDecoderResult r)
        {
            string keyword = r.Keyword;
            if (string.IsNullOrEmpty(keyword)) return;

            // 输入帧 = 10ms; 子采样后帧 = 40ms。
            // timestamps 是子采样后帧的相对索引 (相对 _startFrameIndex)。
            const int inputFrameShiftMs = 10;
            int subsampledShiftMs = inputFrameShiftMs * _model.SubsamplingFactor;

            long startTimeMsLong;
            if (r.Timestamps.Count > 0)
            {
                startTimeMsLong = _startFrameIndex * inputFrameShiftMs + r.Timestamps[0] * subsampledShiftMs;
            }
            else
            {
                startTimeMsLong = _startFrameIndex * inputFrameShiftMs +
                                  Math.Max(0, r.FrameOffset - 1) * subsampledShiftMs;
            }
            long startTimeMs = Math.Max(0, startTimeMsLong);
            long durationMs = Math.Max(0, (long)r.Tokens.Count * subsampledShiftMs);
            long nowMs = startTimeMsLong;

            // 防抖: 同一关键词在 DebounceMs 内不重复
            if (_lastKeyword == keyword && nowMs - _lastKeywordTimeMs < GetDebounceMs(keyword))
            {
                _logger.LogDebug("Debounced keyword '{Keyword}' at t={Ms}ms", keyword, startTimeMs);
                return;
            }
            _lastKeyword = keyword;
            _lastKeywordTimeMs = nowMs;

            int thresholdPercent = (int)Math.Clamp(MathF.Round(_threshold * 100f), 0f, 100f);
            int scorePercent = (int)Math.Clamp(MathF.Round(r.KeywordProb * 100f), 0f, 100f);

            _pendingEvents.Add(new WakeupEventArgs(
                keyword: keyword,
                score: scorePercent,
                threshold: thresholdPercent,
                durationMs: durationMs,
                startTimeMs: startTimeMs));

            _logger.LogInformation(
                "Wakeword hit: keyword={Keyword} score={Score} threshold={Threshold} t={StartTimeMs}ms",
                keyword, scorePercent, thresholdPercent, startTimeMs);
        }

        private int GetDebounceMs(string keyword) =>
            _keywordDebounceMs.TryGetValue(keyword, out int ms) ? ms : _options.DebounceMs;

        /// <summary>
        /// 把命中路径上的 token id 渲染成可读字符串, 供 Near-miss 日志展示 "实际听到了什么"。
        /// 例如 [zh, ōng, g, uó] 会被拼成 "zh ōng g uó"。
        /// </summary>
        private string RenderHeardTokens(long[]? ids)
        {
            if (ids == null || ids.Length == 0) return string.Empty;
            var parts = new string[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                long id = ids[i];
                parts[i] = id < 0 ? "<s>" : _symbols.GetSymbol((int)id);
            }
            return string.Join(' ', parts);
        }

        /// <summary>
        /// 与 sherpa-onnx C++ 的 KeywordSpotterImpl::Reset 等价: 重置 hyps + encoder state,
        /// 不重置 fbank 与 num_processed_frames 的累计计数 (只是把基线推进)。
        /// 命中关键词后 / 长时间静音后调用, 使每段话语作为独立单元解码。
        /// </summary>
        private void ResetDecoderOnly()
        {
            _startFrameIndex += _numProcessedFrames;
            _numProcessedFrames = 0;
            _decoderResult = _decoder.GetEmptyResult();
            foreach (var h in _decoderResult.Hyps.Values) h.ContextState = _contextGraph.Root;
            // 重要: 与 sherpa-onnx 一致, KWS Reset 会清空 encoder 的 KV cache.
            // 不重置 encoder state 会让上一句的声学上下文影响下一次唤醒判定。
            _encoderState = _model.CreateInitialEncoderState();
        }

        private void ResetInternal()
        {
            _fbank.Reset();
            _encoderState = _model.CreateInitialEncoderState();
            _decoderResult = _decoder.GetEmptyResult();
            foreach (var h in _decoderResult.Hyps.Values) h.ContextState = _contextGraph.Root;
            _numProcessedFrames = 0;
            _startFrameIndex = 0;
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
