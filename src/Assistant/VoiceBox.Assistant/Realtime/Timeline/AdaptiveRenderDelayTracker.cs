using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Realtime.Timeline;

internal sealed class AdaptiveRenderDelayTracker
{
    internal const int SampleRate = 16000;

    private const int RequiredConsistentCoarseObservations = 3;

    // Half a DTD hop (320 samples). A hint-only path that disagrees with a
    // stream-local measurement by at least this much has been feeding the model
    // materially misaligned render and is corrected immediately.
    private const double UnverifiedCorrectionThresholdSamples = 160;

    private readonly VoiceAssistantRenderAlignmentOptions _options;

    private readonly double _maximumDelaySamples;

    private readonly double _pathChangeSamples;

    private readonly long _minimumUpdateSamples;

    private bool _hasEstimate;

    private double _delaySamples;

    private double _driftPpm;

    private double _confidence;

    private long _estimatePosition;

    private long _lastObservationPosition;

    private long _lastCorrectionPosition;

    private double _lastObservedDelay;

    private long _generation;

    private int _coarseObservationCount;

    private double _coarseCandidateDelay;

    private double _coarseCandidateConfidence;

    private long _lastCoarseObservationPosition;

    private int _pathChangeCandidateCount;

    private double _pathChangeCandidateDelay;

    private double _pathChangeCandidateConfidence;

    private long _lastPathChangeCandidatePosition;

    // True once the adopted delay path is corroborated by stream-local measurement
    // (AEC3 estimate or correlation fallback). External primary-AEC hints can
    // bootstrap alignment but must never keep a wrong path sticky.
    private bool _pathVerified;

    public long Generation => _generation;

    public AdaptiveRenderDelayTracker(VoiceAssistantRenderAlignmentOptions options)
    {
        _options = options ?? throw new ArgumentNullException("options");
        _maximumDelaySamples = (double)options.MaxTrackedDelayMs * 16.0;
        _pathChangeSamples = (double)options.PathChangeThresholdMs * 16.0;
        _minimumUpdateSamples = Math.Max(1L, checked((long)Math.Round((double)options.DelayUpdateIntervalMs * 16.0)));
    }

    public RenderAlignmentUpdate Advance(long samplePosition)
    {
        if (samplePosition < 0)
        {
            throw new ArgumentOutOfRangeException("samplePosition");
        }
        double delaySamples = PredictDelay(samplePosition);
        return Snapshot(delaySamples, samplePosition);
    }

    public RenderAlignmentUpdate Observe(long samplePosition, in RenderDelayEstimate estimate)
    {
        if (samplePosition < 0)
        {
            throw new ArgumentOutOfRangeException("samplePosition");
        }
        if (!estimate.HasEstimate)
        {
            return Advance(samplePosition);
        }
        if (!double.IsFinite(estimate.DelaySamples) || estimate.DelaySamples < 0.0)
        {
            throw new ArgumentOutOfRangeException("estimate", "Delay estimate must be finite and non-negative.");
        }
        if (!double.IsFinite(estimate.Confidence) || estimate.Confidence < 0.0 || estimate.Confidence > 1.0)
        {
            throw new ArgumentOutOfRangeException("estimate", "Confidence must be finite and in [0, 1].");
        }
        double promotedDelay = Math.Clamp(estimate.DelaySamples, 0.0, _maximumDelaySamples);
        double promotedConfidence = estimate.Confidence;
        bool measured = !estimate.IsExternalHint;
        if (estimate.IsRefined || estimate.IsCorrelationFallback)
        {
            ResetCoarseCandidate();
        }
        else if (!TryPromoteCoarseEstimate(samplePosition, promotedDelay, promotedConfidence, out promotedDelay, out promotedConfidence))
        {
            return Advance(samplePosition);
        }
        if (!_hasEstimate)
        {
            _hasEstimate = true;
            _delaySamples = promotedDelay;
            _estimatePosition = samplePosition;
            _lastObservationPosition = samplePosition;
            _lastCorrectionPosition = samplePosition;
            _lastObservedDelay = promotedDelay;
            _confidence = promotedConfidence;
            _pathVerified = measured;
            return Snapshot(_delaySamples, samplePosition);
        }
        double num = PredictDelay(samplePosition);
        double num2 = promotedDelay - num;
        if (!_pathVerified && measured)
        {
            // The current path came only from external hints. The first stream-local
            // measurement is authoritative: adopt it at once instead of slewing
            // toward it for seconds; a wrong hint path must not enjoy consensus
            // stickiness. When the hint path was materially misaligned the model has
            // been reading the wrong render audio, so a path change restarts it and
            // clears the evidence gathered under misalignment. Full corpus replay
            // showed the model recovers from a clean restart far better than from
            // continuing on state built from misaligned render.
            bool materialJump = Math.Abs(num2) >= UnverifiedCorrectionThresholdSamples;
            _delaySamples = promotedDelay;
            _estimatePosition = samplePosition;
            _lastObservationPosition = samplePosition;
            _lastCorrectionPosition = samplePosition;
            _lastObservedDelay = promotedDelay;
            _driftPpm = 0.0;
            _confidence = Math.Clamp(promotedConfidence, 0.0, 1.0);
            _pathVerified = true;
            ResetCoarseCandidate();
            ResetPathChangeCandidate();
            if (materialJump)
            {
                checked
                {
                    _generation++;
                }
                return Snapshot(_delaySamples, samplePosition) with
                {
                    PathChanged = true
                };
            }
            return Snapshot(_delaySamples, samplePosition);
        }
        checked
        {
            if (promotedConfidence >= _options.MinimumAlignmentConfidence && Math.Abs(num2) >= _pathChangeSamples)
            {
                if (!TryPromotePathChange(
                    samplePosition,
                    promotedDelay,
                    promotedConfidence,
                    out double stableDelay,
                    out double stableConfidence))
                {
                    _delaySamples = num;
                    _estimatePosition = samplePosition;
                    _lastObservationPosition = samplePosition;
                    _confidence = Math.Clamp(_confidence * 0.9 + promotedConfidence * 0.1, 0.0, 1.0);
                    return Snapshot(_delaySamples, samplePosition);
                }

                _delaySamples = stableDelay;
                _estimatePosition = samplePosition;
                _lastObservationPosition = samplePosition;
                _lastCorrectionPosition = samplePosition;
                _lastObservedDelay = stableDelay;
                _driftPpm = 0.0;
                _confidence = stableConfidence;
                _pathVerified = measured;
                _generation++;
                return Snapshot(_delaySamples, samplePosition) with
                {
                    PathChanged = true
                };
            }
        }
        long num3 = samplePosition - _lastObservationPosition;
        if (num3 > 0)
        {
            double value = (promotedDelay - _lastObservedDelay) / (double)num3 * 1000000.0;
            value = Math.Clamp(value, 0.0 - _options.MaximumDriftPpm, _options.MaximumDriftPpm);
            double num4 = 0.08 + 0.22 * promotedConfidence;
            _driftPpm += (value - _driftPpm) * num4;
        }
        if (samplePosition - _lastCorrectionPosition >= _minimumUpdateSamples || promotedConfidence > _confidence)
        {
            long num5 = Math.Max(1L, samplePosition - _estimatePosition);
            double num6 = Math.Max(0.25, _options.MaximumDelaySlewPerSecond * (double)num5 / 16000.0);
            double num7 = 0.15 + 0.35 * promotedConfidence;
            double num8 = Math.Clamp(num2 * num7, 0.0 - num6, num6);
            _delaySamples = Math.Clamp(num + num8, 0.0, _maximumDelaySamples);
            _estimatePosition = samplePosition;
            _lastCorrectionPosition = samplePosition;
        }
        else
        {
            _delaySamples = num;
            _estimatePosition = samplePosition;
        }
        _lastObservationPosition = samplePosition;
        _lastObservedDelay = promotedDelay;
        _confidence = Math.Clamp(_confidence * 0.8 + promotedConfidence * 0.2, 0.0, 1.0);
        if (measured)
            _pathVerified = true;
        ResetPathChangeCandidate();
        return Snapshot(_delaySamples, samplePosition);
    }

    public void Reset()
    {
        _hasEstimate = false;
        _delaySamples = 0.0;
        _driftPpm = 0.0;
        _confidence = 0.0;
        _estimatePosition = 0L;
        _lastObservationPosition = 0L;
        _lastCorrectionPosition = 0L;
        _lastObservedDelay = 0.0;
        _pathVerified = false;
        checked
        {
            _generation++;
            ResetCoarseCandidate();
            ResetPathChangeCandidate();
        }
    }

    private bool TryPromoteCoarseEstimate(long samplePosition, double measuredDelay, double confidence, out double promotedDelay, out double promotedConfidence)
    {
        long num = checked((long)Math.Round((double)_options.EstimateHoldoverMs * 16.0));
        double num2 = Math.Max(1.0, _pathChangeSamples);
        if (_coarseObservationCount == 0 || samplePosition <= _lastCoarseObservationPosition || samplePosition - _lastCoarseObservationPosition > num || Math.Abs(measuredDelay - _coarseCandidateDelay) > num2)
        {
            _coarseObservationCount = 1;
            _coarseCandidateDelay = measuredDelay;
            _coarseCandidateConfidence = confidence;
        }
        else
        {
            _coarseObservationCount++;
            double num3 = 1.0 / (double)_coarseObservationCount;
            _coarseCandidateDelay += (measuredDelay - _coarseCandidateDelay) * num3;
            _coarseCandidateConfidence += (confidence - _coarseCandidateConfidence) * num3;
        }
        _lastCoarseObservationPosition = samplePosition;
        if (_coarseObservationCount < 3)
        {
            promotedDelay = 0.0;
            promotedConfidence = 0.0;
            return false;
        }
        promotedDelay = _coarseCandidateDelay;
        promotedConfidence = _coarseCandidateConfidence;
        ResetCoarseCandidate();
        return true;
    }

    private void ResetCoarseCandidate()
    {
        _coarseObservationCount = 0;
        _coarseCandidateDelay = 0.0;
        _coarseCandidateConfidence = 0.0;
        _lastCoarseObservationPosition = 0L;
    }

    private bool TryPromotePathChange(
        long samplePosition,
        double measuredDelay,
        double confidence,
        out double promotedDelay,
        out double promotedConfidence)
    {
        long holdoverSamples = checked((long)Math.Max(1, _options.EstimateHoldoverMs) * SampleRate / 1000);
        if (_pathChangeCandidateCount == 0 ||
            samplePosition <= _lastPathChangeCandidatePosition ||
            samplePosition - _lastPathChangeCandidatePosition > holdoverSamples ||
            Math.Abs(measuredDelay - _pathChangeCandidateDelay) > Math.Max(1.0, _pathChangeSamples))
        {
            _pathChangeCandidateCount = 1;
            _pathChangeCandidateDelay = measuredDelay;
            _pathChangeCandidateConfidence = confidence;
        }
        else
        {
            _pathChangeCandidateCount++;
            double weight = 1.0 / _pathChangeCandidateCount;
            _pathChangeCandidateDelay +=
                (measuredDelay - _pathChangeCandidateDelay) * weight;
            _pathChangeCandidateConfidence +=
                (confidence - _pathChangeCandidateConfidence) * weight;
        }
        _lastPathChangeCandidatePosition = samplePosition;

        if (_pathChangeCandidateCount < RequiredConsistentCoarseObservations)
        {
            promotedDelay = 0;
            promotedConfidence = 0;
            return false;
        }

        promotedDelay = _pathChangeCandidateDelay;
        promotedConfidence = _pathChangeCandidateConfidence;
        ResetPathChangeCandidate();
        return true;
    }

    private void ResetPathChangeCandidate()
    {
        _pathChangeCandidateCount = 0;
        _pathChangeCandidateDelay = 0.0;
        _pathChangeCandidateConfidence = 0.0;
        _lastPathChangeCandidatePosition = 0L;
    }

    private double PredictDelay(long samplePosition)
    {
        if (!_hasEstimate)
        {
            return 0.0;
        }
        long num = Math.Max(0L, samplePosition - _estimatePosition);
        double value = _delaySamples + (double)num * _driftPpm / 1000000.0;
        return Math.Clamp(value, 0.0, _maximumDelaySamples);
    }

    private RenderAlignmentUpdate Snapshot(double delaySamples, long samplePosition)
    {
        double num = (_hasEstimate ? ((double)Math.Max(0L, samplePosition - _lastObservationPosition) * 1000.0 / 16000.0) : double.PositiveInfinity);
        double num2 = ((num <= (double)_options.EstimateHoldoverMs) ? _confidence : (_confidence * Math.Max(0.0, 1.0 - (num - (double)_options.EstimateHoldoverMs) / (double)_options.EstimateHoldoverMs)));
        return new RenderAlignmentUpdate(_hasEstimate && num2 >= _options.MinimumAlignmentConfidence, PathChanged: false, delaySamples, _driftPpm, num2, _generation);
    }
}
