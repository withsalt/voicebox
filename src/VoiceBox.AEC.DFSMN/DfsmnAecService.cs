using Microsoft.Extensions.Options;
using VoiceBox.AEC.DFSMN.Internal;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.AEC.DFSMN
{
    /// <summary>
    /// Streaming DFSMN acoustic echo cancellation service.
    /// </summary>
    public sealed class DfsmnAecService :
        IAcousticEchoCancellationService,
        IAcousticProcessingMetricsSource
    {
        private readonly object _lock = new();
        private readonly DfsmnAecPipeline _pipeline;
        private bool _disposed;

        public int FrameSize => _pipeline.HopLength;

        public int DelaySamples => _pipeline.DelaySamples;

        public DfsmnAecService(IOptions<DfsmnAecOptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);
            DfsmnAecOptions value = options.Value
                ?? throw new ArgumentException("DFSMN AEC options value cannot be null.", nameof(options));
            _pipeline = new DfsmnAecPipeline(value);
        }

        internal DfsmnAecService(IOptions<DfsmnAecOptions> options, IDfsmnMaskRunner runner, MvnTransform mvn)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(runner);
            ArgumentNullException.ThrowIfNull(mvn);

            DfsmnAecOptions value = options.Value
                ?? throw new ArgumentException("DFSMN AEC options value cannot be null.", nameof(options));
            _pipeline = new DfsmnAecPipeline(value, runner, mvn);
        }

        public void Process(Span<float> capture, ReadOnlySpan<float> render, uint channels, uint sampleRate)
        {
            lock (_lock)
            {
                if (_disposed) return;
                _pipeline.Process(capture, render, channels, sampleRate);
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _pipeline.Reset();
            }
        }

        public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    metrics = default;
                    return false;
                }

                metrics = _pipeline.Metrics;
                return metrics.TimestampTicks != 0;
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _pipeline.Dispose();
            }
        }
    }
}
