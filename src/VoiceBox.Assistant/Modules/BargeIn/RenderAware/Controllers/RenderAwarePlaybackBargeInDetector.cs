using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Estimation;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Policies;
using VoiceBox.Assistant.Realtime.Timeline;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DTD.RenderAware;
using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Controllers;

internal sealed class RenderAwarePlaybackBargeInDetector : IPlaybackBargeInDetector, IDisposable
{
    private readonly record struct PendingDtdObservation(
        long StartSample,
        long EndSampleExclusive,
        bool DecisionEnabled,
        bool EchoLearningEnabled,
        bool IsNearEndSpeech,
        float Probability,
        float SmoothedProbability,
        double CaptureRms,
        double RenderRms,
        double RenderCorrelation,
        double AlignmentConfidence,
        long StreamEpoch,
        long StartSourceSample,
        long EndSourceSampleExclusive,
        bool HasDiscontinuity,
        double DelaySamples);

    private const int TargetSampleRate = 16000;

    private const double SignificantPlaybackGainChangeDb = 6.0;
    private const double GuardedEchoLearningMinimumCorrelation = 0.45;
    private const int AcousticPathChangeRearmFrames = 5;

    /// <summary>Full decay for frames with echo evidence or an active path-change guard.</summary>
    private const double CausalEchoRejectDecay = 1.0;

    /// <summary>Mild decay for frames without near-end evidence, so brief speech gaps do not erase the score.</summary>
    private const double EvidenceAbsentRejectDecay = 0.35;

    private readonly object _sync = new object();

    private readonly IRenderAwareDtdService _dtd;

    private readonly IRenderDelayEstimator? _delayEstimator;

    private readonly AdaptiveRenderDelayTracker _delayTracker;

    private readonly BoundedFractionalRenderBuffer _renderBuffer;

    private readonly BoundedFractionalRenderBuffer _postAecBuffer;

    private readonly TimelineMonoResampler _postAecResampler = new TimelineMonoResampler();

    private readonly BargeInDecisionAccumulator _decisions;

    private readonly RuntimeEchoProfile _echoProfile = new RuntimeEchoProfile();

    private readonly RenderAwareDtdEvidencePolicy _evidencePolicy;

    private readonly TimelinePairResampler _resampler = new TimelinePairResampler();

    private readonly TimelinePairResampler.ResampledPairHandler _resampledPairHandler;

    private readonly float[] _microphoneHop = new float[320];

    private readonly float[] _renderHop = new float[320];

    private readonly float[] _postAecHop = new float[320];

    private readonly float[] _renderEvidenceHop = new float[320];

    private readonly PendingDtdObservation[] _pendingObservations;

    private readonly ILogger _logger;

    private readonly TimeProvider _timeProvider;

    // Alignment (external delay hint + refined-estimate holdover + selection priority) is owned
    // by its own pipeline stage; the detector drives it from its own push path so the model call
    // order stays deterministic.
    private readonly RenderAlignmentStage _alignmentStage;

    private readonly long _postAecWaitSamples;
    private readonly long _acousticPathChangeGuardSamples;

    private uint _inputSampleRate;

    private uint _inputChannels;

    private long _streamEpoch;

    private long _nextTargetSample;

    private long _sourceOriginSample;

    private long _expectedSourceSample;

    private int _hopFill;

    private bool _sourceTimelineInitialized;

    private bool _hopContainsGap;

    private bool _disposed;

    private bool _faulted;

    private Exception? _faultException;

    private bool _hasObservation;

    private int _pendingObservationHead;

    private int _pendingObservationCount;

    private bool _postAecTimelineSeen;

    // Bounded aggregate diagnostics for offline validation and field telemetry. These
    // counters never retain audio or participate in the decision path.
    private long _observationCount;

    private long _nearEndObservationCount;

    private long _postAecEvidenceCount;

    private long _echoLikeObservationCount;

    private double _peakCaptureRms;

    private double _peakPostAecRms;

    private double _minimumPostAecResidualRatio = double.PositiveInfinity;

    private double _maximumAbsoluteRenderCorrelation;

    private double _lastPostAecResidualRatio = double.NaN;

    private double _lastRenderCorrelation = double.NaN;

    private double _lastPostAecRenderCorrelation = double.NaN;

    private bool _lastObservationEchoLike;

    private long _localDelayEstimateCount;

    private long _localRefinedEstimateCount;

    private double _lastLocalDelayMs = double.NaN;

    private double _lastSelectedDelayMs = double.NaN;

    private bool _lastSelectedDelayWasLocal;

    private long _correlationFallbackCount;

    private double _lastCorrelationFallbackDelayMs = double.NaN;

    private double _minimumCorrelationFallbackDelayMs = double.PositiveInfinity;

    private double _maximumCorrelationFallbackDelayMs = double.NegativeInfinity;

    private float _probability;

    private float _smoothedProbability;

    private bool _nearEndSpeech;

    private RenderAlignmentUpdate _alignment;

    private RenderTimelineStatus _renderStatus = RenderTimelineStatus.Unavailable;

    private long _lastStartSample;

    private long _lastEndSample;

    private int _pushHopCount;

    private int _pushObservationCount;

    private bool _pushStateReset;

    private bool _pushDecisionEnabled;

    private bool _pushEchoLearningEnabled;

    private bool _pushInferenceEnabled;

    private bool _pushRenderValid;

    private bool _decisionDiscontinuityPending;

    private RenderSignalTimeline _lastRenderTimeline;

    private RuntimeEchoProfileSnapshot _lastEchoProfile;

    private long _trustedEchoObservationCount;

    private long _preservedNearEndObservationCount;

    private bool _lastObservationPreservedNearEnd;

    private bool _hasPlaybackGain;
    private float _lastPlaybackGain = 1f;
    private long _acousticPathChangeGuardUntilSample;
    private bool _acousticPathChangeLatched;
    private int _acousticPathChangeClearFrames;
    private long _acousticPathChangeCount;
    private bool _lastObservationAcousticPathChange;
    private bool _lastObservationPathChangeGuarded;
    private string _lastAcousticPathChangeReason = "none";

    internal RenderAwarePlaybackBargeInDetector(VoiceAssistantRenderAwareDtdOptions dtdOptions, VoiceAssistantRenderAlignmentOptions alignmentOptions, IRenderAwareDtdService dtd, IRenderDelayEstimator? delayEstimator = null, ILogger? logger = null, TimeProvider? timeProvider = null)
    {
        ArgumentNullException.ThrowIfNull(dtdOptions, "dtdOptions");
        ArgumentNullException.ThrowIfNull(alignmentOptions, "alignmentOptions");
        ArgumentNullException.ThrowIfNull(dtdOptions.Model, "dtdOptions.Model");
        _dtd = dtd ?? throw new ArgumentNullException("dtd");
        if (dtd.SampleRate != 16000 || dtd.ChannelCount != 1 || dtd.HopLength != 320)
        {
            throw new ArgumentException("The playback DTD must use the render-aware model format.", "dtd");
        }
        _delayEstimator = delayEstimator;
        _delayTracker = new AdaptiveRenderDelayTracker(alignmentOptions);
        int val;
        checked
        {
            val = unchecked(checked((alignmentOptions.MaxTrackedDelayMs + 120) * 16000) / 1000) + dtd.WindowLength + 320;
        }
        _renderBuffer = new BoundedFractionalRenderBuffer(Math.Max(val, dtd.WindowLength + 320));
        _postAecBuffer = new BoundedFractionalRenderBuffer(Math.Max(val, dtd.WindowLength + 320));
        int capacity = Math.Max(1, (dtdOptions.DecisionHistoryMs + 19) / 20);
        _decisions = new BargeInDecisionAccumulator(capacity);
        _evidencePolicy = new RenderAwareDtdEvidencePolicy(dtdOptions.Model.Threshold);
        _pendingObservations = new PendingDtdObservation[Math.Max(16, (int)Math.Ceiling((alignmentOptions.MaxTrackedDelayMs + 500) / 20.0) + 16)];
        _resampledPairHandler = OnResampledPair;
        _logger = logger ?? NullLogger.Instance;
        _timeProvider = timeProvider ?? TimeProvider.System;
        _alignmentStage = new RenderAlignmentStage(
            _timeProvider,
            alignmentOptions.DelayHintMaxAgeMs,
            checked((long)Math.Max(1, alignmentOptions.EstimateHoldoverMs) * 16));
        _postAecWaitSamples = checked((long)Math.Max(dtd.WindowLength + 320, (alignmentOptions.MaxTrackedDelayMs + 500) * 16));
        if (dtdOptions.AcousticPathChangeGuardMs is < 0 or > 60000)
            throw new ArgumentOutOfRangeException(nameof(dtdOptions.AcousticPathChangeGuardMs));
        _acousticPathChangeGuardSamples = checked((long)dtdOptions.AcousticPathChangeGuardMs * 16);
    }

    public DtdProcessBatch Push(
        in DuplexAudioFrame frame,
        RenderDelayHint? primaryAecDelay,
        bool decisionEnabled,
        bool echoLearningEnabled = true)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_faulted)
            {
                throw new InvalidOperationException(
                    "Render-aware DTD detector is faulted and must be recreated.",
                    _faultException);
            }
            _pushHopCount = 0;
            _pushObservationCount = 0;
            _pushStateReset = false;
            _alignmentStage.DelayHint = _alignmentStage.IsFreshDelayHint(primaryAecDelay, frame.TimestampTicks)
                ? primaryAecDelay
                : null;
            try
            {
                _pushDecisionEnabled = decisionEnabled;
                _pushEchoLearningEnabled = echoLearningEnabled;
                _pushInferenceEnabled = true;
                _pushRenderValid = false;
                CaptureSignalTimeline microphoneTimeline = frame.CaptureBeforeAecTimeline;
                ReadOnlyMemory<float> microphone = frame.CaptureBeforeAecValid;
                if (!microphoneTimeline.IsValid || microphone.IsEmpty)
                {
                    microphoneTimeline = frame.CaptureRawTimeline;
                    microphone = frame.CaptureRawValid;
                }
                RenderTimelineStatus renderStatus = ResolveRenderStatus(in frame);
                if (!CanProcess(in frame, microphone, in microphoneTimeline))
                {
                    ResetPipeline();
                    _renderStatus = renderStatus;
                    _pushStateReset = true;
                    return Batch();
                }
                CaptureRange sourceRange = microphoneTimeline.SourceRange;
                bool formatChanged = _inputSampleRate != 0 &&
                    (_inputSampleRate != frame.SampleRate || _inputChannels != frame.Channels);
                bool epochChanged = _streamEpoch != 0L && _streamEpoch != sourceRange.StreamEpoch;
                bool captureDiscontinuous = _sourceTimelineInitialized &&
                    sourceRange.StartSample != _expectedSourceSample;
                bool renderDiscontinuous = HasRenderDiscontinuity(in frame);
                if (formatChanged || epochChanged || captureDiscontinuous ||
                    frame.HasDiscontinuity || renderDiscontinuous)
                {
                    ResetPipeline();
                    _renderStatus = renderStatus;
                    _pushStateReset = true;
                    return Batch();
                }
                if (renderStatus is RenderTimelineStatus.Missing or RenderTimelineStatus.Gap)
                {
                    ResetPipeline();
                    _renderStatus = renderStatus;
                    _pushStateReset = true;
                    return Batch();
                }
                _inputSampleRate = frame.SampleRate;
                _inputChannels = frame.Channels;
                _streamEpoch = sourceRange.StreamEpoch;
                if (!_sourceTimelineInitialized)
                {
                    _sourceTimelineInitialized = true;
                    _sourceOriginSample = sourceRange.StartSample;
                }
                _expectedSourceSample = sourceRange.EndSampleExclusive;
                _renderStatus = renderStatus;
                ObservePlaybackGain(frame.PlaybackGain);
                if (frame.RenderTimeline.IsValid)
                {
                    _lastRenderTimeline = frame.RenderTimeline;
                }
                checked
                {
                    ReadOnlySpan<float> span;
                    ReadOnlyMemory<float> renderReference;
                    _pushRenderValid = true;
                    try
                    {
                        AppendPostAecEvidence(in frame);
                        int length = (int)((ulong)frame.FrameCount * frame.Channels);
                        span = microphone.Span;
                        renderReference = frame.RenderReference;
                        _resampler.Append(
                            span.Slice(0, length),
                            renderReference.Span.Slice(0, length),
                            frame.Channels,
                            frame.SampleRate,
                            _resampledPairHandler);
                        ResolvePendingObservations(forceFallback: false);
                        return Batch();
                    }
                    catch (Exception exception)
                    {
                        MarkFaulted(exception);
                        ResetPipeline();
                        _pushStateReset = true;
                        throw;
                    }
                }
            }
            finally
            {
                _alignmentStage.DelayHint = null;
            }
        }
    }

    public BargeInDecision GetRecentDecision(TimeSpan window, int minDurationMs, float minRms)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return (window <= TimeSpan.Zero) ? default(BargeInDecision) : _decisions.AggregateRecent(window.TotalMilliseconds, (minDurationMs > 0) ? minDurationMs : 280, minRms, BargeInDecisionParameters.Default);
        }
    }

    public RenderAwareBargeInSnapshot GetSnapshot()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long remainingGuardSamples = Math.Max(
                0,
                _acousticPathChangeGuardUntilSample - _nextTargetSample);
            return new RenderAwareBargeInSnapshot(
                _hasObservation,
                _probability,
                _smoothedProbability,
                _nearEndSpeech,
                _alignment.IsAligned,
                _alignment.DelaySamples * 1000.0 / 16000.0,
                _alignment.DriftPpm,
                _alignment.Confidence,
                _alignment.Generation,
                _streamEpoch,
                _lastStartSample,
                _lastEndSample,
                _renderStatus,
                _faulted,
                _observationCount,
                _nearEndObservationCount,
                _postAecEvidenceCount,
                _echoLikeObservationCount,
                _peakCaptureRms,
                _peakPostAecRms,
                _minimumPostAecResidualRatio,
                _maximumAbsoluteRenderCorrelation,
                _lastPostAecResidualRatio,
                _lastRenderCorrelation,
                _lastPostAecRenderCorrelation,
                _lastObservationEchoLike,
                _localDelayEstimateCount,
                _localRefinedEstimateCount,
                _lastLocalDelayMs,
                _lastSelectedDelayMs,
                _lastSelectedDelayWasLocal,
                _correlationFallbackCount,
                _lastCorrelationFallbackDelayMs,
                double.IsFinite(_minimumCorrelationFallbackDelayMs) ? _minimumCorrelationFallbackDelayMs : double.NaN,
                double.IsFinite(_maximumCorrelationFallbackDelayMs) ? _maximumCorrelationFallbackDelayMs : double.NaN,
                _lastEchoProfile.Reliability,
                _lastEchoProfile.TrustedEchoDurationMs,
                _lastEchoProfile.RelevantEchoDurationMs,
                _lastEchoProfile.PredictedCaptureRms,
                _lastEchoProfile.PredictedPostAecRms,
                _lastEchoProfile.PredictedResidualRatio,
                _trustedEchoObservationCount,
                _preservedNearEndObservationCount,
                _lastObservationPreservedNearEnd,
                remainingGuardSamples > 0,
                remainingGuardSamples * 1000.0 / TargetSampleRate,
                _acousticPathChangeCount,
                _lastObservationAcousticPathChange,
                _lastObservationPathChangeGuarded,
                _lastAcousticPathChangeReason);
        }
    }

    public void ResetDecisionState()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _decisions.Reset();
        }
    }

    public void ResetStream()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ResetPipeline();
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (!_disposed)
            {
                _disposed = true;
                _delayEstimator?.Dispose();
                _dtd.Dispose();
                _renderBuffer.Reset();
                _decisions.Reset();
                ResetEchoProfile();
            }
        }
    }

    private static bool CanProcess(in DuplexAudioFrame frame, ReadOnlyMemory<float> microphone, in CaptureSignalTimeline microphoneTimeline)
    {
        if (frame.SampleRate == 0 || frame.Channels == 0 || frame.FrameCount == 0)
        {
            return false;
        }
        ulong num = (ulong)frame.FrameCount * (ulong)frame.Channels;
        if (num > int.MaxValue)
        {
            return false;
        }
        if (frame.RenderReference.Length < (int)num ||
            !microphoneTimeline.IsValid ||
            microphone.Length < (int)num)
        {
            return false;
        }

        return !frame.RenderTimeline.IsValid || frame.RenderTimeline.Length == (long)num;
    }

    private bool HasRenderDiscontinuity(in DuplexAudioFrame frame)
    {
        RenderSignalTimeline next = frame.RenderTimeline;
        return next.IsValid && (next.HasDiscontinuity || (_lastRenderTimeline.IsValid && !_lastRenderTimeline.IsContiguousWith(in next)));
    }

    private static RenderTimelineStatus ResolveRenderStatus(in DuplexAudioFrame frame)
    {
        if (frame.RenderTimeline.IsValid)
        {
            return frame.RenderTimeline.Status;
        }
        if (frame.RenderTimeline != default(RenderSignalTimeline))
        {
            return RenderTimelineStatus.Missing;
        }
        return (!frame.RenderReference.IsEmpty) ? RenderTimelineStatus.Valid : RenderTimelineStatus.Missing;
    }

    private void OnResampledPair(float microphone, float render)
    {
        _renderBuffer.Append(render, _nextTargetSample, _pushRenderValid);
        _microphoneHop[_hopFill] = microphone;
        _renderHop[_hopFill] = render;
        _hopContainsGap |= !_pushInferenceEnabled;
        _hopFill++;
        _nextTargetSample++;
        if (_hopFill != 320)
        {
            return;
        }
        _pushHopCount++;
        long num = _nextTargetSample - 320;
        _hopFill = 0;
        if (_hopContainsGap)
        {
            _hopContainsGap = false;
            return;
        }
        RenderDelayEstimate localEstimate = _delayEstimator?.Process(_renderHop, _microphoneHop) ?? default(RenderDelayEstimate);
        if (localEstimate.HasEstimate)
        {
            _localDelayEstimateCount++;
            _lastLocalDelayMs = localEstimate.DelaySamples * 1000.0 / 16000.0;
            if (localEstimate.IsRefined)
            {
                _localRefinedEstimateCount++;
                _alignmentStage.LastRefinedEstimateSample = num;
            }
            else if (localEstimate.IsCorrelationFallback)
            {
                _correlationFallbackCount++;
                _lastCorrelationFallbackDelayMs = localEstimate.DelaySamples * 1000.0 / 16000.0;
                _minimumCorrelationFallbackDelayMs = Math.Min(
                    _minimumCorrelationFallbackDelayMs,
                    _lastCorrelationFallbackDelayMs);
                _maximumCorrelationFallbackDelayMs = Math.Max(
                    _maximumCorrelationFallbackDelayMs,
                    _lastCorrelationFallbackDelayMs);
            }
        }
        RenderDelayEstimate estimate = _alignmentStage.SelectDelayEstimate(localEstimate, num);
        _lastSelectedDelayMs = estimate.HasEstimate
            ? estimate.DelaySamples * 1000.0 / 16000.0
            : double.NaN;
        _lastSelectedDelayWasLocal = localEstimate.HasEstimate &&
            ((localEstimate.IsRefined && estimate.IsRefined) ||
             Math.Abs(estimate.DelaySamples - localEstimate.DelaySamples) < 1e-9);
        _alignment = (estimate.HasEstimate ? _delayTracker.Observe(num, in estimate) : _delayTracker.Advance(num));
        if (_alignment.PathChanged)
        {
            ResetModelForPathChange();
        }
        else
        {
            if (!_alignment.IsAligned)
            {
                return;
            }
            double startSample = (double)num - _alignment.DelaySamples;
            if (!_renderBuffer.TryRead(startSample, _renderHop, out bool intersectsRenderGap))
            {
                if (intersectsRenderGap)
                    ResetModelForPathChange();
                return;
            }
            RenderAwareDtdObservation? renderAwareDtdObservation = _dtd.Process(_microphoneHop, _renderHop, 1u, 16000u);
            if (!renderAwareDtdObservation.HasValue)
            {
                return;
            }
            RenderAwareDtdObservation valueOrDefault = renderAwareDtdObservation.GetValueOrDefault();
            _pushObservationCount++;
            _hasObservation = true;
            _probability = valueOrDefault.Probability;
            _smoothedProbability = valueOrDefault.SmoothedProbability;
            _nearEndSpeech = valueOrDefault.IsNearEndSpeech;
            _observationCount++;
            if (valueOrDefault.IsNearEndSpeech)
                _nearEndObservationCount++;
            // One pass over the hop pair yields both RMS values and their correlation.
            // Each accumulator still sums in ascending sample order, so the results are
            // bit-identical to the three separate loops this replaces.
            RmsPairWithCorrelation(
                _microphoneHop,
                _renderHop,
                out double captureRms,
                out double renderRms,
                out double renderCorrelation);
            _peakCaptureRms = Math.Max(_peakCaptureRms, captureRms);
            _maximumAbsoluteRenderCorrelation = Math.Max(
                _maximumAbsoluteRenderCorrelation,
                Math.Abs(renderCorrelation));
            long num2 = MapTargetToSource(num);
            long num3 = MapTargetToSource(num + 320);
            _lastStartSample = num2;
            _lastEndSample = num3;
            if (_pushDecisionEnabled || _pushEchoLearningEnabled)
            {
                bool decisionDiscontinuityPending =
                    _pushDecisionEnabled && _decisionDiscontinuityPending;
                if (_pushDecisionEnabled)
                    _decisionDiscontinuityPending = false;
                EnqueuePendingObservation(new PendingDtdObservation(
                    num,
                    num + 320,
                    _pushDecisionEnabled,
                    _pushEchoLearningEnabled,
                    valueOrDefault.IsNearEndSpeech,
                    valueOrDefault.Probability,
                    valueOrDefault.SmoothedProbability,
                    captureRms,
                    renderRms,
                    renderCorrelation,
                    _alignment.Confidence,
                    _streamEpoch,
                    num2,
                    num3,
                    decisionDiscontinuityPending,
                    _alignment.DelaySamples));
                ResolvePendingObservations(forceFallback: false);
            }
        }
    }

    private void AppendPostAecEvidence(in DuplexAudioFrame frame)
    {
        CaptureSignalTimeline timeline = frame.CaptureAfterAecTimeline;
        ReadOnlyMemory<float> audio = frame.CaptureAfterAecValid;
        if (!timeline.IsValid || audio.IsEmpty)
            return;

        _postAecResampler.Append(
            in timeline,
            frame.CaptureAfterAec,
            frame.SampleRate,
            frame.Channels,
            _sourceOriginSample,
            _streamEpoch,
            _postAecBuffer);
        _postAecTimelineSeen = true;
    }

    private void EnqueuePendingObservation(in PendingDtdObservation observation)
    {
        if (_pendingObservationCount == _pendingObservations.Length)
            ResolvePendingObservations(forceFallback: true);

        if (_pendingObservationCount == _pendingObservations.Length)
        {
            // The queue is bounded by the maximum tracked delay. Losing the oldest
            // item is preferable to retaining unbounded callback state; resolve it
            // as model-only evidence before overwriting the slot.
            ResolveOnePendingObservation(hasPostAecEvidence: false, forceFallback: true);
        }

        int index = (_pendingObservationHead + _pendingObservationCount) % _pendingObservations.Length;
        _pendingObservations[index] = observation;
        _pendingObservationCount++;
    }

    private void ResolvePendingObservations(bool forceFallback)
    {
        while (_pendingObservationCount > 0)
        {
            PendingDtdObservation observation = _pendingObservations[_pendingObservationHead];
            bool hasPostAecEvidence = _postAecTimelineSeen &&
                _postAecBuffer.TryRead(observation.StartSample, _postAecHop);
            bool expired = forceFallback ||
                (!_postAecTimelineSeen && _nextTargetSample - observation.EndSampleExclusive >= _postAecWaitSamples) ||
                (_postAecTimelineSeen && _postAecBuffer.Count > 0 &&
                    _postAecBuffer.StartSample > observation.StartSample);
            if (!hasPostAecEvidence && !expired)
                break;

            ResolveOnePendingObservation(hasPostAecEvidence, forceFallback: expired);
        }
    }

    private void ResolveOnePendingObservation(bool hasPostAecEvidence, bool forceFallback)
    {
        PendingDtdObservation observation = _pendingObservations[_pendingObservationHead];
        _pendingObservations[_pendingObservationHead] = default;
        _pendingObservationHead = (_pendingObservationHead + 1) % _pendingObservations.Length;
        _pendingObservationCount--;

        double postAecRms = double.NaN;
        double residualRatio = double.NaN;
        double postAecRenderCorrelation = double.NaN;
        if (hasPostAecEvidence)
        {
            postAecRms = Rms(_postAecHop);
            double inputRms = observation.CaptureRms;
            residualRatio = inputRms > 1e-4 ? postAecRms / inputRms : double.PositiveInfinity;
            _postAecEvidenceCount++;
            _peakPostAecRms = Math.Max(_peakPostAecRms, postAecRms);
            _minimumPostAecResidualRatio = Math.Min(_minimumPostAecResidualRatio, residualRatio);
            _lastPostAecResidualRatio = residualRatio;
            _lastRenderCorrelation = observation.RenderCorrelation;
            if (_renderBuffer.TryRead(
                observation.StartSample - observation.DelaySamples,
                _renderEvidenceHop))
            {
                postAecRenderCorrelation = Correlation(_postAecHop, _renderEvidenceHop);
            }
            _lastPostAecRenderCorrelation = postAecRenderCorrelation;
        }
        else
        {
            _lastPostAecResidualRatio = double.NaN;
            _lastRenderCorrelation = observation.RenderCorrelation;
            _lastPostAecRenderCorrelation = double.NaN;
        }

        _echoProfile.Advance(20.0);
        RuntimeEchoProfileSnapshot profile = _echoProfile.Snapshot(observation.RenderRms);
        var evidence = new RenderAwareDtdFrameEvidence(
            observation.DecisionEnabled,
            observation.EchoLearningEnabled,
            observation.IsNearEndSpeech,
            observation.Probability,
            observation.SmoothedProbability,
            observation.CaptureRms,
            observation.RenderRms,
            observation.RenderCorrelation,
            observation.AlignmentConfidence,
            hasPostAecEvidence,
            postAecRms,
            residualRatio,
            postAecRenderCorrelation);
        RenderAwareDtdEvidenceDecision evidenceDecision =
            _evidencePolicy.Evaluate(in evidence, in profile);
        UpdateAcousticPathChangeState(
            evidenceDecision.AcousticPathChange,
            observation.StartSample);
        bool pathChangeGuarded = IsAcousticPathChangeGuardActiveAt(observation.StartSample);
        _lastObservationEchoLike = evidenceDecision.EchoLike;
        _lastObservationPreservedNearEnd = evidenceDecision.PreservedNearEnd;
        _lastObservationAcousticPathChange = evidenceDecision.AcousticPathChange;
        _lastObservationPathChangeGuarded = pathChangeGuarded;
        if (evidenceDecision.EchoLike)
            _echoLikeObservationCount++;
        if (evidenceDecision.PreservedNearEnd)
            _preservedNearEndObservationCount++;

        if (observation.DecisionEnabled)
        {
            string reason = pathChangeGuarded
                ? "acoustic-path-change-guard"
                : forceFallback && !hasPostAecEvidence
                ? "render-aware-dtd-no-post-aec-evidence"
                : evidenceDecision.Reason;
            // A rejected frame either carries positive causal echo evidence or it merely
            // failed to carry near-end evidence. Only the former justifies erasing
            // previously earned near-end duration in full. The path-change guard
            // deliberately suppresses confirmation while the acoustic path is unsettled, so
            // it keeps full decay as well.
            double rejectDecay = evidenceDecision.EchoLike || pathChangeGuarded
                ? CausalEchoRejectDecay
                : EvidenceAbsentRejectDecay;
            _decisions.Add(new NearEndFrameDecision(
                0L,
                20.0,
                evidenceDecision.Accepted && !pathChangeGuarded,
                evidenceDecision.EchoLike,
                RenderActive: true,
                observation.CaptureRms,
                0.0,
                rejectDecay,
                reason,
                observation.StreamEpoch,
                observation.StartSourceSample,
                observation.EndSourceSampleExclusive,
                observation.HasDiscontinuity,
                PostAecResidualRatio: hasPostAecEvidence ? residualRatio : double.NaN,
                RenderCorrelation: observation.RenderCorrelation,
                PostAecRenderCorrelation: postAecRenderCorrelation));
        }

        double strongestCorrelation = Math.Max(
            Math.Abs(observation.RenderCorrelation),
            double.IsFinite(postAecRenderCorrelation)
                ? Math.Abs(postAecRenderCorrelation)
                : 0);
        bool guardedEchoLearning = pathChangeGuarded &&
            observation.EchoLearningEnabled &&
            hasPostAecEvidence &&
            observation.RenderRms > 1e-5 &&
            observation.CaptureRms > 1e-4 &&
            strongestCorrelation >= GuardedEchoLearningMinimumCorrelation;
        if (evidenceDecision.TrustedEcho || guardedEchoLearning)
        {
            _echoProfile.ObserveTrustedEcho(
                observation.RenderRms,
                observation.CaptureRms,
                postAecRms,
                residualRatio,
                observation.Probability,
                observation.SmoothedProbability,
                20.0);
            _trustedEchoObservationCount++;
            _lastEchoProfile = _echoProfile.Snapshot(observation.RenderRms);
        }
        else
        {
            // Nothing between the pre-policy snapshot and here mutated the profile, and
            // the query RMS is the same, so re-snapshotting would reproduce `profile`
            // bit for bit. Most frames take this path.
            _lastEchoProfile = profile;
        }
    }

    private void ObservePlaybackGain(float playbackGain)
    {
        if (!float.IsFinite(playbackGain) || playbackGain < 0)
            return;

        float current = Math.Clamp(playbackGain, 0f, 1f);
        if (!_hasPlaybackGain)
        {
            _hasPlaybackGain = true;
            _lastPlaybackGain = current;
            return;
        }

        float previous = _lastPlaybackGain;
        _lastPlaybackGain = current;
        // Attenuation is the first stage of an acoustic-only interruption. Its render reference
        // already includes the lower gain; keep the near-end evidence that caused the duck.
        // Restoring/increasing gain can expose a stronger echo and still needs the guard below.
        if (current <= previous)
            return;
        bool crossedSilenceBoundary = (previous <= 1e-4f || current <= 1e-4f) &&
            Math.Abs(current - previous) >= 0.05f;
        double changeDb = previous > 1e-4f && current > 1e-4f
            ? Math.Abs(20.0 * Math.Log10(current / previous))
            : 0;
        if (crossedSilenceBoundary || changeDb >= SignificantPlaybackGainChangeDb)
            StartAcousticPathChangeGuard(_nextTargetSample, "playback-gain-change");
    }

    private void UpdateAcousticPathChangeState(bool acousticPathChange, long startSample)
    {
        if (acousticPathChange)
        {
            _acousticPathChangeClearFrames = 0;
            if (!_acousticPathChangeLatched)
            {
                _acousticPathChangeLatched = true;
                StartAcousticPathChangeGuard(startSample, "coherent-echo-envelope-break");
            }
            return;
        }

        if (!_acousticPathChangeLatched)
            return;

        if (++_acousticPathChangeClearFrames >= AcousticPathChangeRearmFrames)
        {
            _acousticPathChangeLatched = false;
            _acousticPathChangeClearFrames = 0;
        }
    }

    private void StartAcousticPathChangeGuard(long startSample, string reason)
    {
        if (_acousticPathChangeGuardSamples <= 0 ||
            IsAcousticPathChangeGuardActiveAt(startSample))
        {
            return;
        }

        _acousticPathChangeGuardUntilSample = checked(
            startSample + _acousticPathChangeGuardSamples);
        _acousticPathChangeCount++;
        _lastAcousticPathChangeReason = reason;
        _decisions.Reset();
    }

    private bool IsAcousticPathChangeGuardActiveAt(long samplePosition)
        => _acousticPathChangeGuardSamples > 0 &&
            samplePosition < _acousticPathChangeGuardUntilSample;

    private void ClearPendingObservations()
    {
        Array.Clear(_pendingObservations);
        _pendingObservationHead = 0;
        _pendingObservationCount = 0;
    }

    private void MarkFaulted(Exception exception)
    {
        _faulted = true;
        _faultException ??= exception;
        _logger.LogError(exception, "Render-aware DTD detector entered a terminal fault state.");
    }

    private long MapTargetToSource(long targetSample)
    {
        double a = (double)targetSample * ((double)_inputSampleRate * (double)_inputChannels / 16000.0);
        return checked(_sourceOriginSample + (long)Math.Round(a));
    }

    private void ResetModelForPathChange()
    {
        _dtd.Reset();
        _decisions.Reset();
        ClearPendingObservations();
        ResetEchoProfile();
        _hasObservation = false;
        _probability = 0f;
        _smoothedProbability = 0f;
        _nearEndSpeech = false;
        _decisionDiscontinuityPending = true;
        _pushStateReset = true;
    }

    private DtdProcessBatch Batch()
    {
        return new DtdProcessBatch(_pushHopCount, _pushObservationCount, _pushObservationCount > 0, _probability, _smoothedProbability, _nearEndSpeech, _pushStateReset, _renderStatus);
    }

    private void ResetPipeline()
    {
        _dtd.Reset();
        _delayEstimator?.Reset();
        _delayTracker.Reset();
        _renderBuffer.Reset();
        _postAecBuffer.Reset();
        _postAecResampler.Reset();
        _resampler.Reset();
        _decisions.Reset();
        ClearPendingObservations();
        ResetEchoProfile();
        _inputSampleRate = 0u;
        _inputChannels = 0u;
        _streamEpoch = 0L;
        _nextTargetSample = 0L;
        _sourceOriginSample = 0L;
        _expectedSourceSample = 0L;
        _hopFill = 0;
        _sourceTimelineInitialized = false;
        _hopContainsGap = false;
        _pushRenderValid = false;
        _hasObservation = false;
        _probability = 0f;
        _smoothedProbability = 0f;
        _nearEndSpeech = false;
        _alignment = default(RenderAlignmentUpdate);
        _lastStartSample = 0L;
        _lastEndSample = 0L;
        _lastRenderTimeline = default(RenderSignalTimeline);
        _postAecTimelineSeen = false;
        _renderStatus = RenderTimelineStatus.Unavailable;
        _decisionDiscontinuityPending = true;
        _observationCount = 0L;
        _nearEndObservationCount = 0L;
        _postAecEvidenceCount = 0L;
        _echoLikeObservationCount = 0L;
        _peakCaptureRms = 0d;
        _peakPostAecRms = 0d;
        _minimumPostAecResidualRatio = double.PositiveInfinity;
        _maximumAbsoluteRenderCorrelation = 0d;
        _lastPostAecResidualRatio = double.NaN;
        _lastRenderCorrelation = double.NaN;
        _lastPostAecRenderCorrelation = double.NaN;
        _lastObservationEchoLike = false;
        _hasPlaybackGain = false;
        _lastPlaybackGain = 1f;
        _acousticPathChangeGuardUntilSample = 0;
        _acousticPathChangeLatched = false;
        _acousticPathChangeClearFrames = 0;
        _acousticPathChangeCount = 0;
        _lastObservationAcousticPathChange = false;
        _lastObservationPathChangeGuarded = false;
        _lastAcousticPathChangeReason = "none";
        _localDelayEstimateCount = 0L;
        _localRefinedEstimateCount = 0L;
        _lastLocalDelayMs = double.NaN;
        _lastSelectedDelayMs = double.NaN;
        _lastSelectedDelayWasLocal = false;
        _alignmentStage.LastRefinedEstimateSample = -1;
        _correlationFallbackCount = 0L;
        _lastCorrelationFallbackDelayMs = double.NaN;
        _minimumCorrelationFallbackDelayMs = double.PositiveInfinity;
        _maximumCorrelationFallbackDelayMs = double.NegativeInfinity;
    }

    private void ResetEchoProfile()
    {
        _echoProfile.Reset();
        _lastEchoProfile = default;
        _trustedEchoObservationCount = 0;
        _preservedNearEndObservationCount = 0;
        _lastObservationPreservedNearEnd = false;
    }

    private static double Rms(ReadOnlySpan<float> samples)
    {
        double num = 0.0;
        for (int i = 0; i < samples.Length; i++)
        {
            num += (double)(samples[i] * samples[i]);
        }
        return Math.Sqrt(num / (double)samples.Length);
    }

    /// <summary>
    /// Computes both hop RMS values and their normalized correlation in one pass over the
    /// pair, replacing three separate traversals. Each accumulator keeps the exact
    /// expression and ascending summation order of the loop it came from, so all three
    /// results are bit-identical to <see cref="Rms"/>/<see cref="Correlation"/>.
    /// </summary>
    /// <remarks>
    /// The RMS and correlation energies are deliberately accumulated separately: RMS
    /// rounds each square to float before widening, while correlation multiplies in
    /// double. Sharing one accumulator would shift results by a rounding step.
    /// </remarks>
    private static void RmsPairWithCorrelation(
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        out double leftRms,
        out double rightRms,
        out double correlation)
    {
        int count = left.Length;
        if (right.Length != count)
            throw new ArgumentException("Hop spans must have equal length.", nameof(right));

        double leftRmsEnergy = 0;
        double rightRmsEnergy = 0;
        double leftEnergy = 0;
        double rightEnergy = 0;
        double dot = 0;
        for (int i = 0; i < count; i++)
        {
            float leftSample = left[i];
            float rightSample = right[i];
            leftRmsEnergy += (double)(leftSample * leftSample);
            rightRmsEnergy += (double)(rightSample * rightSample);
            double a = leftSample;
            double b = rightSample;
            leftEnergy += a * a;
            rightEnergy += b * b;
            dot += a * b;
        }

        leftRms = Math.Sqrt(leftRmsEnergy / count);
        rightRms = Math.Sqrt(rightRmsEnergy / count);
        double denominator = Math.Sqrt(leftEnergy * rightEnergy);
        correlation = count > 0 && denominator > 1e-12
            ? Math.Clamp(dot / denominator, -1.0, 1.0)
            : 0;
    }

    private static double Correlation(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        int count = Math.Min(left.Length, right.Length);
        if (count <= 0)
            return 0;

        double leftEnergy = 0;
        double rightEnergy = 0;
        double dot = 0;
        for (int i = 0; i < count; i++)
        {
            double a = left[i];
            double b = right[i];
            leftEnergy += a * a;
            rightEnergy += b * b;
            dot += a * b;
        }

        double denominator = Math.Sqrt(leftEnergy * rightEnergy);
        return denominator > 1e-12 ? Math.Clamp(dot / denominator, -1.0, 1.0) : 0;
    }

    DtdProcessBatch IPlaybackBargeInDetector.Push(
        in DuplexAudioFrame frame,
        RenderDelayHint? primaryAecDelay,
        bool decisionEnabled,
        bool echoLearningEnabled)
    {
        return Push(in frame, primaryAecDelay, decisionEnabled, echoLearningEnabled);
    }
}
