using Microsoft.Extensions.Options;
using VoiceBox.AEC.DTLN.Internal;
using VoiceBox.AEC.DTLN.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.AEC.DTLN
{
    /// <summary>
    /// Streaming DTLN-aec acoustic echo cancellation service.
    ///
    /// Pipeline:
    /// capture/render FIFO -> optional render delay -> DTLN stage 1 (spectral mask) ->
    /// DTLN stage 2 (time-domain refinement) -> processed capture FIFO.
    /// </summary>
    public sealed class DtlnAecService :
        IAcousticEchoCancellationService,
        IAcousticProcessingMetricsSource
    {
        private readonly DtlnAecOptions _options;
        private readonly FloatFifo _captureFifo;
        private readonly FloatFifo _renderFifo;
        private readonly FloatFifo _outputFifo;
        private readonly DynamicDelayLine _referenceDelay;
        private readonly IDelayEstimator? _delayEstimator;
        private readonly DtlnAecProcessor _processor;
        private readonly int _algorithmicDelaySamples;

        private readonly float[] _captureHop = new float[DtlnAecOptions.HopLength];
        private readonly float[] _renderHop = new float[DtlnAecOptions.HopLength];
        private readonly float[] _delayedRenderHop = new float[DtlnAecOptions.HopLength];
        private readonly float[] _processedHop = new float[DtlnAecOptions.HopLength];

        private AcousticProcessingMetrics _metrics;
        // 0 means no metrics yet; odd values mark an in-progress write.
        private long _metricsSequence;
        private bool _unsupportedLogged;
        private bool _disposed;

        public int FrameSize => DtlnAecOptions.HopLength;

        public int DelaySamples => _algorithmicDelaySamples;

        public DtlnAecService(IOptions<DtlnAecOptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);
            _options = options.Value ?? throw new ArgumentException("DTLN AEC options value cannot be null.", nameof(options));
            _options.Validate();
            _algorithmicDelaySamples = _options.OutputPrerollSamples;

            int fifoCapacity = Math.Max(DtlnAecOptions.BlockLength * 8, 4096);
            _captureFifo = new FloatFifo(fifoCapacity);
            _renderFifo = new FloatFifo(fifoCapacity);
            _outputFifo = new FloatFifo(fifoCapacity);
            _outputFifo.WriteZeros(_algorithmicDelaySamples);

            int maxDelaySamples = _options.EnableAutoDelayEstimation
                ? _options.MaxAutoDelayMs * (int)_options.SampleRate / 1000
                : _options.ReferenceDelaySamples;
            _referenceDelay = new DynamicDelayLine(_options.ReferenceDelaySamples, maxDelaySamples);
            if (_options.EnableAutoDelayEstimation)
            {
                _delayEstimator = new WebRtcAec3DelayAdapter((int)_options.SampleRate);
            }
            (string part1, string part2) = ResolveModelPaths(_options);
            DtlnAecOnnxRunner? runner = null;
            try
            {
                runner = new DtlnAecOnnxRunner(part1, part2, _options);
                _processor = new DtlnAecProcessor(runner, _options.WarmupHops);
                runner = null;
            }
            finally
            {
                runner?.Dispose();
            }
        }

        public void Process(Span<float> capture, ReadOnlySpan<float> render, uint channels, uint sampleRate)
        {
            if (_disposed || capture.IsEmpty)
                return;

            if (channels != DtlnAecOptions.SupportedChannels ||
                sampleRate != DtlnAecOptions.SupportedSampleRate ||
                render.Length < capture.Length)
            {
                if (!_unsupportedLogged)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"[DtlnAecService] Unsupported callback format: channels={channels}, sampleRate={sampleRate}, capture={capture.Length}, render={render.Length}. Passing through.");
                    _unsupportedLogged = true;
                }
                return;
            }

            ReplaceNonFiniteInPlace(capture);
            _captureFifo.Write(capture);
            _renderFifo.WriteReplacingNonFinite(render[..capture.Length]);

            // 在线延迟估计(WebRTC AEC3):喂入时间对齐的原始 render/capture 对,自动跟踪往返延迟。
            // 估计出新延迟时平滑更新延迟线,无需手工配置 ReferenceDelaySamples。
            if (_delayEstimator is not null)
            {
                int? newDelay = _delayEstimator.FeedFrame(render[..capture.Length], capture);
                if (newDelay.HasValue)
                    _referenceDelay.SetDelay(newDelay.Value);
            }

            bool processedAny = false;
            try
            {
                while (_captureFifo.Count >= DtlnAecOptions.HopLength && _renderFifo.Count >= DtlnAecOptions.HopLength)
                {
                    _captureFifo.Read(_captureHop);
                    _renderFifo.Read(_renderHop);
                    ProcessHop();
                    processedAny = true;
                }
            }
            catch
            {
                ResetAfterProcessingFailure();
                throw;
            }

            if (!processedAny)
                UpdateInputMetrics(capture, render[..capture.Length]);

            int toCopy = Math.Min(_outputFifo.Count, capture.Length);
            if (toCopy > 0)
            {
                _outputFifo.Read(capture[..toCopy]);
            }
        }

        public void Reset()
        {
            if (_disposed) return;

            _captureFifo.Clear();
            _renderFifo.Clear();
            _outputFifo.Clear();
            _outputFifo.WriteZeros(_algorithmicDelaySamples);
            _referenceDelay.Reset();
            _delayEstimator?.Reset();
            _processor.Reset();
            ClearMetrics();
        }

        public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
        {
            if (System.Threading.Volatile.Read(ref _disposed))
            {
                metrics = default;
                return false;
            }

            System.Threading.SpinWait spin = default;
            while (true)
            {
                long before = System.Threading.Volatile.Read(ref _metricsSequence);
                if (before == 0)
                {
                    metrics = default;
                    return false;
                }

                if ((before & 1) != 0)
                {
                    spin.SpinOnce();
                    continue;
                }

                System.Threading.Thread.MemoryBarrier();
                metrics = _metrics;
                System.Threading.Thread.MemoryBarrier();

                long after = System.Threading.Volatile.Read(ref _metricsSequence);
                if (before == after && (after & 1) == 0)
                    return true;

                spin.SpinOnce();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _delayEstimator?.Dispose();
            _processor.Dispose();
        }

        private void ProcessHop()
        {
            _referenceDelay.Process(_renderHop, _delayedRenderHop);
            _processor.ProcessHop(_captureHop, _delayedRenderHop, _processedHop);
            ClipInPlace(_processedHop, _options.ClipLimit);
            _outputFifo.Write(_processedHop);
            UpdateOutputMetrics(_captureHop, _delayedRenderHop, _processedHop);
        }

        private void ResetAfterProcessingFailure()
        {
            _captureFifo.Clear();
            _renderFifo.Clear();
            _outputFifo.Clear();
            _outputFifo.WriteZeros(_algorithmicDelaySamples);
            _referenceDelay.Reset();
            _delayEstimator?.Reset();
            ClearMetrics();

            try
            {
                _processor.Reset();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[DtlnAecService] Failed to reset after processing error: {ex.Message}");
            }
        }

        private void ClearMetrics()
        {
            System.Threading.Volatile.Write(ref _metricsSequence, 1);
            _metrics = default;
            System.Threading.Volatile.Write(ref _metricsSequence, 0);
        }

        private void UpdateInputMetrics(ReadOnlySpan<float> capture, ReadOnlySpan<float> render)
        {
            SignalStats stats = SignalStats.Compute(capture, render, capture);
            PublishMetrics(BuildMetrics(stats, erle: 0));
        }

        private void UpdateOutputMetrics(ReadOnlySpan<float> capture, ReadOnlySpan<float> render, ReadOnlySpan<float> processed)
        {
            SignalStats stats = SignalStats.Compute(capture, render, processed);
            double erle = stats.ProcessedEnergy > 1e-12
                ? 10.0 * Math.Log10((stats.CaptureEnergy + 1e-12) / (stats.ProcessedEnergy + 1e-12))
                : 0.0;
            PublishMetrics(BuildMetrics(stats, erle));
        }

        private void PublishMetrics(AcousticProcessingMetrics metrics)
        {
            long sequence = System.Threading.Volatile.Read(ref _metricsSequence);
            if ((sequence & 1) != 0)
                sequence++;

            System.Threading.Volatile.Write(ref _metricsSequence, sequence + 1);
            System.Threading.Thread.MemoryBarrier();
            _metrics = metrics;
            System.Threading.Volatile.Write(ref _metricsSequence, sequence + 2);
        }

        private AcousticProcessingMetrics BuildMetrics(SignalStats stats, double erle)
        {
            double erl = stats.RenderEnergy > 1e-12
                ? 10.0 * Math.Log10((stats.RenderEnergy + 1e-12) / (stats.CaptureEnergy + 1e-12))
                : 0.0;

            // 延迟可观测性:DelayMs = 延迟线当前实际施加的延迟;DelayMedianMs = 估计器最新(中值)估计值。
            bool hasDelay = _delayEstimator is not null && _delayEstimator.HasEstimate;
            int appliedDelayMs = _referenceDelay.CurrentDelay * 1000 / DtlnAecOptions.SupportedSampleRate;
            int estimatedDelayMs = (_delayEstimator?.LastEstimatedDelay ?? 0) * 1000 / DtlnAecOptions.SupportedSampleRate;

            return new AcousticProcessingMetrics(
                HasEchoReturnLoss: stats.RenderEnergy > 1e-12 && stats.CaptureEnergy > 1e-12,
                EchoReturnLoss: erl,
                HasEchoReturnLossEnhancement: stats.CaptureEnergy > 1e-12,
                EchoReturnLossEnhancement: erle,
                HasDelayMs: hasDelay,
                DelayMs: appliedDelayMs,
                HasDelayMedianMs: hasDelay,
                DelayMedianMs: estimatedDelayMs,
                HasResidualEchoLikelihood: stats.RenderEnergy > 1e-12 && stats.ProcessedEnergy > 1e-12,
                ResidualEchoLikelihood: Math.Abs(stats.RenderProcessedCorrelation),
                TimestampTicks: System.Diagnostics.Stopwatch.GetTimestamp());
        }

        private static void ClipInPlace(Span<float> samples, float limit)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                float sample = samples[i];
                samples[i] = float.IsFinite(sample) ? DspMath.Clamp(sample, -limit, limit) : 0f;
            }
        }

        private static void ReplaceNonFiniteInPlace(Span<float> samples)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                if (!float.IsFinite(samples[i]))
                    samples[i] = 0f;
            }
        }

        private static (string Part1, string Part2) ResolveModelPaths(DtlnAecOptions options)
        {
            if (options.ModelPart1Path is not null && options.ModelPart2Path is not null)
            {
                if (!File.Exists(options.ModelPart1Path))
                    throw new FileNotFoundException($"DTLN AEC part-1 model not found: {options.ModelPart1Path}", options.ModelPart1Path);
                if (!File.Exists(options.ModelPart2Path))
                    throw new FileNotFoundException($"DTLN AEC part-2 model not found: {options.ModelPart2Path}", options.ModelPart2Path);
                return (options.ModelPart1Path, options.ModelPart2Path);
            }

            int units = options.ModelSize.Units();
            string dir = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "aec", "dtln");
            string part1 = Path.Combine(dir, $"dtln_aec_{units}_1.onnx");
            string part2 = Path.Combine(dir, $"dtln_aec_{units}_2.onnx");
            if (!File.Exists(part1) || !File.Exists(part2))
            {
                throw new FileNotFoundException(
                    $"Built-in DTLN AEC models not found under '{dir}'. Set DtlnAecOptions.ModelPart1Path/ModelPart2Path or copy exported models to the runtime directory.",
                    File.Exists(part1) ? part2 : part1);
            }

            return (part1, part2);
        }

        private readonly record struct SignalStats(
            double CaptureEnergy,
            double RenderEnergy,
            double ProcessedEnergy,
            double RenderProcessedCorrelation)
        {
            public static SignalStats Compute(ReadOnlySpan<float> capture, ReadOnlySpan<float> render, ReadOnlySpan<float> processed)
            {
                int length = Math.Min(capture.Length, Math.Min(render.Length, processed.Length));
                if (length <= 0) return default;

                double captureEnergy = 0;
                double renderEnergy = 0;
                double processedEnergy = 0;
                double renderProcessedDot = 0;

                for (int i = 0; i < length; i++)
                {
                    double c = FiniteOrZero(capture[i]);
                    double r = FiniteOrZero(render[i]);
                    double p = FiniteOrZero(processed[i]);
                    captureEnergy += c * c;
                    renderEnergy += r * r;
                    processedEnergy += p * p;
                    renderProcessedDot += r * p;
                }

                double corr = 0;
                double denom = Math.Sqrt(renderEnergy * processedEnergy);
                if (denom > 1e-12)
                    corr = Math.Clamp(renderProcessedDot / denom, -1.0, 1.0);

                return new SignalStats(captureEnergy, renderEnergy, processedEnergy, corr);
            }

            private static float FiniteOrZero(float sample)
                => float.IsFinite(sample) ? sample : 0f;
        }
    }
}
