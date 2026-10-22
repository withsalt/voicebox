using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Diagnostics.Metrics.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

namespace VoiceBox.Assistant.Diagnostics.Metrics
{
    /// <summary>
    /// Throttled reader for the optional acoustic diagnostics port. It holds only owned snapshots
    /// and never references a concrete echo-cancellation service.
    /// </summary>
    internal sealed class AecMetricsSampler
    {
        private const int SampleIntervalMs = 100;

        private readonly IAcousticProcessingDiagnostics? _diagnostics;
        private readonly TimeProvider _timeProvider;
        private readonly ILogger _logger;
        private readonly object _lock = new();
        private long _lastSampleTicks;
        private int _hasSampled;
        private AcousticProcessingMetricsSnapshot _latest;
        private bool _hasLatest;

        public AecMetricsSampler(
            IAcousticProcessingDiagnostics? diagnostics,
            ILogger logger,
            TimeProvider? timeProvider = null)
        {
            _diagnostics = diagnostics;
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _timeProvider = timeProvider ?? TimeProvider.System;
        }

        public void SampleIfDue(long observationTimestampTicks = 0)
        {
            var diagnostics = _diagnostics;
            if (diagnostics is null)
                return;

            long now = observationTimestampTicks > 0
                ? observationTimestampTicks
                : _timeProvider.GetTimestamp();
            long last = Volatile.Read(ref _lastSampleTicks);
            if (Volatile.Read(ref _hasSampled) != 0 &&
                _timeProvider.GetElapsedTime(last, now) < TimeSpan.FromMilliseconds(SampleIntervalMs))
            {
                return;
            }
            if (Interlocked.CompareExchange(ref _lastSampleTicks, now, last) != last)
                return;
            Volatile.Write(ref _hasSampled, 1);

            try
            {
                if (diagnostics.TryGetLatestMetrics(out AcousticProcessingMetricsSnapshot metrics))
                    StoreLatest(metrics, now);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "读取 AEC 统计失败, 已跳过本次采样。");
            }
        }

        public bool TryGetLatest(out AcousticProcessingMetricsSnapshot metrics)
        {
            lock (_lock)
            {
                metrics = _latest;
                return _hasLatest;
            }
        }

        public void Reset()
        {
            Volatile.Write(ref _lastSampleTicks, 0);
            Volatile.Write(ref _hasSampled, 0);
            lock (_lock)
            {
                _latest = default;
                _hasLatest = false;
            }
        }

        private void StoreLatest(
            AcousticProcessingMetricsSnapshot metrics,
            long observationTimestampTicks)
        {
            if (metrics.TimestampTicks <= 0)
                metrics = metrics.WithTimestamp(observationTimestampTicks);

            lock (_lock)
            {
                _latest = metrics;
                _hasLatest = true;
            }
        }
    }
}
