using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Realtime.Timeline;

/// <summary>
/// Alignment stage of the render-aware pipeline. It owns the external delay hint, the refined
/// estimate holdover window, and the priority order used to pick the authoritative delay estimate
/// for one hop.
/// </summary>
/// <remarks>
/// Pure policy: no model call, no I/O and no timer. The detector mutates this stage from its own
/// push path, so the alignment decision stays deterministic for a given frame sequence.
/// </remarks>
internal sealed class RenderAlignmentStage
{
    private readonly TimeProvider _timeProvider;
    private readonly int _delayHintMaxAgeMs;
    private readonly long _refinedEstimateHoldoverSamples;
    private RenderDelayHint? _delayHint;
    private long _lastRefinedEstimateSample = -1;

    public RenderAlignmentStage(
        TimeProvider timeProvider,
        int delayHintMaxAgeMs,
        long refinedEstimateHoldoverSamples)
    {
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
        _delayHintMaxAgeMs = delayHintMaxAgeMs;
        _refinedEstimateHoldoverSamples = refinedEstimateHoldoverSamples;
    }

    public RenderDelayHint? DelayHint
    {
        get => _delayHint;
        set => _delayHint = value;
    }

    public long LastRefinedEstimateSample
    {
        get => _lastRefinedEstimateSample;
        set => _lastRefinedEstimateSample = value;
    }

    public void Reset() => _lastRefinedEstimateSample = -1;

    /// <summary>
    /// Accepts an external AEC delay hint only while it is newer than the configured maximum age
    /// relative to the current frame timestamp.
    /// </summary>
    public bool IsFreshDelayHint(RenderDelayHint? hint, long frameTimestampTicks)
    {
        if (hint is not { IsValid: true } value ||
            value.TimestampTicks <= 0 ||
            frameTimestampTicks <= 0)
        {
            return false;
        }

        TimeSpan elapsed = _timeProvider.GetElapsedTime(value.TimestampTicks, frameTimestampTicks);
        return elapsed >= TimeSpan.Zero && elapsed <= TimeSpan.FromMilliseconds(_delayHintMaxAgeMs);
    }

    public bool HasRecentRefinedEstimate(long samplePosition)
        => _lastRefinedEstimateSample >= 0 &&
            samplePosition >= _lastRefinedEstimateSample &&
            samplePosition - _lastRefinedEstimateSample <= _refinedEstimateHoldoverSamples;

    public RenderDelayEstimate SelectDelayEstimate(
        in RenderDelayEstimate localEstimate,
        long samplePosition)
    {
        // A refined per-stream observation is authoritative.
        if (localEstimate.HasEstimate && localEstimate.IsRefined)
        {
            return localEstimate;
        }

        // The correlation fallback is emitted only after the local estimator has
        // observed a stable candidate repeatedly. Once promoted it is stream-local
        // evidence and may supersede a stale refined holdover when the measured path
        // has changed.
        if (localEstimate.HasEstimate && localEstimate.IsCorrelationFallback)
            return localEstimate;

        // Do not perturb a recent refined path with a lower-quality source. The
        // tracker continues to extrapolate measured drift during speech-heavy
        // intervals where correlation estimators naturally stop reporting.
        if (HasRecentRefinedEstimate(samplePosition))
            return default;

        if (_delayHint is { IsValid: true, IsStable: true } stableHint)
        {
            return new RenderDelayEstimate(
                HasEstimate: true,
                DelaySamples: stableHint.DelayMilliseconds * 16.0,
                Confidence: 0.85,
                IsRefined: true,
                IsCorrelationFallback: false,
                IsExternalHint: true);
        }

        if (localEstimate.HasEstimate)
            return localEstimate;

        if (_delayHint is { IsValid: true } hint)
        {
            return new RenderDelayEstimate(
                HasEstimate: true,
                DelaySamples: hint.DelayMilliseconds * 16.0,
                Confidence: 0.65,
                IsRefined: false,
                IsCorrelationFallback: false,
                IsExternalHint: true);
        }
        return localEstimate;
    }
}
