using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    /// <summary>
    /// Shared streaming DFSMN front end and inference pipeline.
    /// It owns the real-time AEC state and runs the model once per 20 ms hop.
    /// </summary>
    internal sealed class DfsmnAecPipeline : IDisposable
    {
        private const uint NativeSampleRate = 16000;

        private readonly DfsmnAecOptions _options;
        private readonly int _hopLength;
        private readonly int _frameLength;
        private readonly int _featureDim;
        private readonly int _maskBins;

        private readonly FloatFifo _captureFifo;
        private readonly FloatFifo _renderFifo;
        private readonly FloatFifo _outputFifo;
        private readonly AdaptiveNlmsEchoCanceller _linearAec;
        private readonly SampleDelayLine _referenceDelay;
        private readonly DoubleTalkDetector _doubleTalk;
        private readonly ResidualEchoSuppressor? _residualSuppressor;
        private readonly FeatureFrameBuilder _featureBuilder;
        private readonly IDfsmnMaskRunner _maskRunner;
        private readonly MaskedStftProcessor _maskedStft;
        private readonly SlidingFrameBuffer _nearWindow;
        private readonly SlidingFrameBuffer _linearWindow;
        private readonly SlidingFrameBuffer _echoWindow;

        private readonly float[] _captureHop;
        private readonly float[] _renderHop;
        private readonly float[] _delayedRenderHop;
        private readonly float[] _linearHop;
        private readonly float[] _echoHop;
        private readonly float[] _feature;
        private readonly float[] _mask;
        private readonly float[] _processedHop;
        private readonly float[] _vadProbability;
        private readonly NearEndProtectionGate _protectionGate;
        private readonly bool _protectionActive;

        private bool _unsupportedLogged;
        private bool _disposed;
        private bool _modelFreezeNextHop;
        private CaptureRange _captureTimeline;
        private bool _hasCaptureTimeline;
        private AcousticProcessingMetrics _metrics;

        public DfsmnAecPipeline(DfsmnAecOptions options)
            : this(options, maskRunner: null, mvn: null, requireVadOutput: false)
        {
        }

        internal DfsmnAecPipeline(
            DfsmnAecOptions options,
            IDfsmnMaskRunner? maskRunner,
            MvnTransform? mvn,
            bool requireVadOutput = false)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _options.Validate();

            _hopLength = DfsmnAecOptions.DefaultHopLength;
            _frameLength = DfsmnAecOptions.DefaultFeatureFrameLength;
            _featureDim = DfsmnAecOptions.DefaultFeatureDim;
            _maskBins = DfsmnAecOptions.DefaultMaskBins;

            int fifoCapacity = Math.Max(_frameLength * 8, 4096);
            _captureFifo = new FloatFifo(fifoCapacity);
            _renderFifo = new FloatFifo(fifoCapacity);
            _outputFifo = new FloatFifo(fifoCapacity);
            _outputFifo.WriteZeros(_frameLength);

            _linearAec = new AdaptiveNlmsEchoCanceller(
                _options.Linear.FilterLength,
                _options.Linear.AdaptationStep,
                _options.Linear.Epsilon,
                _options.Linear.Leakage);
            _referenceDelay = new SampleDelayLine(_options.ReferenceDelaySamples);
            _doubleTalk = new DoubleTalkDetector(_options.DoubleTalk, _hopLength, (int)NativeSampleRate);
            if (_options.Residual.Enabled)
            {
                _residualSuppressor = new ResidualEchoSuppressor(
                    _options.Residual.EchoSuppressionDb,
                    _options.Residual.EchoSuppressionRatio,
                    _options.Residual.AttackMs,
                    _options.Residual.ReleaseMs,
                    _hopLength,
                    (int)NativeSampleRate);
            }

            _nearWindow = new SlidingFrameBuffer(_frameLength, _hopLength);
            _linearWindow = new SlidingFrameBuffer(_frameLength, _hopLength);
            _echoWindow = new SlidingFrameBuffer(_frameLength, _hopLength);

            _captureHop = new float[_hopLength];
            _renderHop = new float[_hopLength];
            _delayedRenderHop = new float[_hopLength];
            _linearHop = new float[_hopLength];
            _echoHop = new float[_hopLength];
            _feature = new float[_featureDim];
            _mask = new float[_maskBins];
            _processedHop = new float[_hopLength];
            _vadProbability = new float[1];

            MvnTransform effectiveMvn = mvn ?? MvnTransform.Load(DfsmnModelFiles.ResolveMvnPath(_options), _featureDim);
            _featureBuilder = new FeatureFrameBuilder(_frameLength, DfsmnAecOptions.DefaultFeatureBins, effectiveMvn);
            _maskRunner = maskRunner ?? new OnnxDfsmnMaskRunner(DfsmnModelFiles.ResolveModelPath(_options), _options);
            if (_maskRunner.FeatureDim != _featureDim)
                throw new ArgumentException($"DFSMN runner feature dim {_maskRunner.FeatureDim} does not match {_featureDim}.", nameof(maskRunner));
            if (_maskRunner.MaskBins != _maskBins)
                throw new ArgumentException($"DFSMN runner mask bins {_maskRunner.MaskBins} does not match {_maskBins}.", nameof(maskRunner));
            if (requireVadOutput && !_maskRunner.HasVadProbabilities)
            {
                throw new InvalidOperationException(
                    "The configured DFSMN model does not expose a near-end VAD probability output. " +
                    "Use the mask+VAD export for DfsmnAecVadService, or register DfsmnAecService for AEC-only processing.");
            }
            _maskedStft = new MaskedStftProcessor(_frameLength, _hopLength, _options.MaskFloorDb);
            // 近端保护依赖逐 hop vad_prob; mask-only 模型上自动退化为旧行为。
            _protectionGate = new NearEndProtectionGate(
                _options.NearEndProtection,
                _options.MaskFloorDb,
                _hopLength,
                (int)NativeSampleRate);
            _protectionActive = _options.NearEndProtection.Enabled && _maskRunner.HasVadProbabilities;
        }

        public int HopLength => _hopLength;
        public int FrameLength => _frameLength;
        public int DelaySamples => _frameLength;
        public AcousticProcessingMetrics Metrics => _metrics;

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate,
            DfsmnPipelineHopHandler? hopHandler = null)
            => Process(
                capture,
                render,
                channels,
                sampleRate,
                default,
                hopHandler);

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate,
            in CaptureRange captureRange,
            DfsmnPipelineHopHandler? hopHandler = null)
        {
            if (_disposed || capture.IsEmpty)
                return;

            if (channels != 1 || sampleRate != NativeSampleRate || render.Length < capture.Length)
            {
                if (!_unsupportedLogged)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"[DfsmnAecPipeline] Unsupported callback format: channels={channels}, sampleRate={sampleRate}, capture={capture.Length}, render={render.Length}. Passing through.");
                    _unsupportedLogged = true;
                }
                return;
            }

            bool hasUsableCaptureRange =
                captureRange.IsValid &&
                captureRange.Length == capture.Length;
            if (_captureFifo.Count > 0)
            {
                bool timelineChanged =
                    _hasCaptureTimeline != hasUsableCaptureRange ||
                    (_hasCaptureTimeline &&
                     !_captureTimeline.IsContiguousWith(in captureRange));
                if (timelineChanged)
                {
                    // A model hop must never combine samples from different stream epochs
                    // or from timeline-aware and legacy processing modes.
                    Reset();
                }
            }

            if (hasUsableCaptureRange)
            {
                if (_captureFifo.Count == 0)
                {
                    _captureTimeline = captureRange;
                    _hasCaptureTimeline = true;
                }
                else
                {
                    _captureTimeline = _captureTimeline with
                    {
                        EndSequenceNumber = captureRange.EndSequenceNumber,
                        EndSampleExclusive = captureRange.EndSampleExclusive,
                    };
                }
            }
            else
            {
                _captureTimeline = default;
                _hasCaptureTimeline = false;
            }

            _captureFifo.Write(capture);
            _renderFifo.Write(render[..capture.Length]);

            while (_captureFifo.Count >= _hopLength && _renderFifo.Count >= _hopLength)
            {
                _captureFifo.Read(_captureHop);
                _renderFifo.Read(_renderHop);
                SanitizeInPlace(_captureHop);
                SanitizeInPlace(_renderHop);
                CaptureRange hopRange = _hasCaptureTimeline
                    ? SliceRange(in _captureTimeline, 0, _hopLength)
                    : default;
                ProcessHop(hopHandler, in hopRange);
                if (_hasCaptureTimeline)
                {
                    _captureTimeline = _captureTimeline with
                    {
                        StartSample = checked(_captureTimeline.StartSample + _hopLength),
                        StartSequenceNumber = _captureTimeline.EndSequenceNumber,
                    };
                    if (_captureTimeline.StartSample >= _captureTimeline.EndSampleExclusive)
                    {
                        _captureTimeline = default;
                        _hasCaptureTimeline = false;
                    }
                }
            }

            int toCopy = Math.Min(_outputFifo.Count, capture.Length);
            if (toCopy > 0)
                _outputFifo.Read(capture[..toCopy]);
        }

        public void Reset()
        {
            if (_disposed) return;

            _captureFifo.Clear();
            _renderFifo.Clear();
            _captureTimeline = default;
            _hasCaptureTimeline = false;
            _outputFifo.Clear();
            _outputFifo.WriteZeros(_frameLength);
            _linearAec.Reset();
            _referenceDelay.Reset();
            _doubleTalk.Reset();
            _residualSuppressor?.Reset();
            _nearWindow.Reset();
            _linearWindow.Reset();
            _echoWindow.Reset();
            _maskRunner.Reset();
            _maskedStft.Reset();
            _protectionGate.Reset();
            _modelFreezeNextHop = false;
            _metrics = default;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _maskRunner.Dispose();
        }

        private void ProcessHop(
            DfsmnPipelineHopHandler? hopHandler,
            in CaptureRange captureRange)
        {
            _referenceDelay.Process(_renderHop, _delayedRenderHop);

            // 能量 DTD 之外叠加模型冻结: DTD 要求麦克风比预期回声高 ~6dB, 大声外放下的
            // 轻声叠说达不到; 上一 hop 的近端概率 (20ms 滞后, 含保护包络 hangover) 把这类
            // 双讲也冻住, 防止 NLMS 把用户语音当滤波误差吸收。提示传入 DTD 内部, 同时
            // 保护其耦合估计不被叠说能量污染; DTD 被显式关闭时模型冻结仍独立生效。
            // (_modelFreezeNextHop 仅由 _protectionActive 的写入端置位, 此处无需再门控。)
            bool modelNearEnd = _modelFreezeNextHop;
            bool freeze = _doubleTalk.IsDoubleTalk(_captureHop, _delayedRenderHop, modelNearEnd) || modelNearEnd;
            _linearAec.Process(_captureHop, _delayedRenderHop, _linearHop, _echoHop, adapt: !freeze);

            bool nearReady = _nearWindow.Push(_captureHop);
            bool linearReady = _linearWindow.Push(_linearHop);
            bool echoReady = _echoWindow.Push(_echoHop);
            if (!nearReady || !linearReady || !echoReady)
                return;

            _featureBuilder.Build(_nearWindow.Frame, _linearWindow.Frame, _echoWindow.Frame, _feature);
            // 近端保护需要逐 hop vad_prob (与 mask 同一次推理), 即使没有 VAD 事件订阅者。
            if (hopHandler is null && !_protectionActive)
            {
                _maskRunner.Run(_feature, 1, _mask);
                _vadProbability[0] = 0f;
            }
            else
            {
                if (!_maskRunner.HasVadProbabilities)
                {
                    throw new InvalidOperationException(
                        "The configured DFSMN runner cannot produce near-end VAD probabilities.");
                }
                _maskRunner.Run(_feature, 1, _mask, _vadProbability);
            }

            // 模型确认近端叠说时: 抬高掩码下限 (限制频域最大抑制)、收起标量残差刀、
            // 并为下一 hop 冻结线性自适应。纯回声 hop (envelope=0) 与旧行为逐位一致。
            float envelope = _protectionActive ? _protectionGate.Observe(_vadProbability[0]) : 0f;
            _modelFreezeNextHop = _protectionActive && _protectionGate.ShouldFreezeAdaptation(_vadProbability[0]);

            _maskedStft.ProcessFrame(_linearWindow.Frame, _mask, _processedHop, _protectionGate.MaskFloor(envelope));
            _residualSuppressor?.Process(_echoHop, _processedHop, _protectionGate.ResidualGainFloor(envelope));
            ClipInPlace(_processedHop, _options.ClipLimit);
            _outputFifo.Write(_processedHop);
            UpdateMetrics(captureRange.TimestampTicks);

            hopHandler?.Invoke(
                _captureHop,
                _linearHop,
                _processedHop,
                _vadProbability[0],
                in captureRange);
        }

        private static CaptureRange SliceRange(
            in CaptureRange sourceRange,
            int offset,
            int length)
        {
            if (!sourceRange.IsValid ||
                offset < 0 ||
                length <= 0 ||
                offset > sourceRange.Length - length)
            {
                return default;
            }

            long start = checked(sourceRange.StartSample + offset);
            long end = checked(start + length);
            return new CaptureRange(
                sourceRange.StreamEpoch,
                sourceRange.StartSequenceNumber,
                sourceRange.EndSequenceNumber,
                start,
                end,
                sourceRange.TimestampTicks);
        }

        private void UpdateMetrics(long observationTimestampTicks)
        {
            _metrics = new AcousticProcessingMetrics(
                HasEchoReturnLoss: false,
                EchoReturnLoss: 0,
                HasEchoReturnLossEnhancement: false,
                EchoReturnLossEnhancement: 0,
                HasDelayMs: _options.ReferenceDelaySamples > 0,
                DelayMs: (int)Math.Round(1000.0 * _options.ReferenceDelaySamples / NativeSampleRate),
                HasDelayMedianMs: false,
                DelayMedianMs: 0,
                HasResidualEchoLikelihood: true,
                ResidualEchoLikelihood: EstimateResidualEchoLikelihood(),
                TimestampTicks: observationTimestampTicks > 0
                    ? observationTimestampTicks
                    : System.Diagnostics.Stopwatch.GetTimestamp(),
                HasProcessedCaptureRms: true,
                ProcessedCaptureRms: Rms(_processedHop),
                HasInputCaptureRms: true,
                InputCaptureRms: Rms(_captureHop));
        }

        private double EstimateResidualEchoLikelihood()
        {
            double echo = Energy(_echoHop);
            double output = Energy(_processedHop);
            if (echo <= 1e-12) return 0;
            return Math.Clamp(echo / (echo + output + 1e-12), 0, 1);
        }

        private static double Energy(ReadOnlySpan<float> samples)
        {
            double acc = 0;
            for (int i = 0; i < samples.Length; i++)
                acc += (double)samples[i] * samples[i];
            return acc;
        }

        private static double Rms(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty)
                return 0;
            return Math.Sqrt(Energy(samples) / samples.Length);
        }

        private static void SanitizeInPlace(Span<float> samples)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                if (!float.IsFinite(samples[i]))
                    samples[i] = 0f;
            }
        }

        private static void ClipInPlace(Span<float> samples, float limit)
        {
            for (int i = 0; i < samples.Length; i++)
                samples[i] = DspMath.Clamp(samples[i], -limit, limit);
        }
    }
}
