using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

/// <summary>
/// Owner-speaker authorization for a pending wake event. The candidate interval is the exact KWS
/// focus plus configured lead-in/post-trigger context on the identity timeline; evidence usually
/// already sits in the prefetch cache, so the first evaluation after the keyword decides
/// immediately whenever a first complete window has been scored. Speaker evidence is mandatory:
/// KWS-enabled compositions cannot be built without it.
/// </summary>
internal sealed class WakeupSpeakerAuthorization
{
    private readonly ISpeakerEvidenceProvider _speakerEvidence;
    private readonly TimeProvider _timeProvider;
    private readonly int _strongScoreMargin;
    private readonly int _postTriggerEvidenceSamples;
    private long _pendingReadyTimestamp = -1;
    private SpeakerCandidateId _candidateId;
    private SpeakerVerificationResult _lastResult;

    public WakeupSpeakerAuthorization(
        ISpeakerEvidenceProvider speakerEvidence,
        TimeProvider timeProvider,
        int strongScoreMargin,
        int postTriggerEvidenceSamples)
    {
        _speakerEvidence = speakerEvidence ?? throw new ArgumentNullException(nameof(speakerEvidence));
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
        if (strongScoreMargin is < 0 or > 100)
            throw new ArgumentOutOfRangeException(nameof(strongScoreMargin));
        if (postTriggerEvidenceSamples < 0)
            throw new ArgumentOutOfRangeException(nameof(postTriggerEvidenceSamples));
        _strongScoreMargin = strongScoreMargin;
        _postTriggerEvidenceSamples = postTriggerEvidenceSamples;
    }

    /// <summary>The last speaker verdict observed for the current candidate, for diagnostics.</summary>
    public SpeakerVerificationResult LastResult => _lastResult;

    public SpeakerCandidateId CandidateId => _candidateId;

    public void Reset()
    {
        _pendingReadyTimestamp = -1;
        if (!_candidateId.IsNone)
            _speakerEvidence.Cancel(_candidateId);
        _candidateId = default;
        _lastResult = default;
    }

    public WakeupAuthorizationOutcome Authorize(
        WakeupEventArgs wakeup,
        WakeupPreRollTimeline timeline,
        long captureEndSample,
        long timestampTicks)
    {
        long wakeStartSample;
        try
        {
            wakeStartSample = timeline.ComputeWakeStartCaptureSample(wakeup);
        }
        catch (OverflowException)
        {
            return WakeupAuthorizationOutcome.Unavailable;
        }

        if (wakeStartSample < timeline.PreRollStartCaptureSample)
            return WakeupAuthorizationOutcome.Unavailable;

        long wakeEndSample;
        try
        {
            wakeEndSample = timeline.ComputeWakeEndCaptureSample(wakeStartSample, wakeup);
        }
        catch (OverflowException)
        {
            return WakeupAuthorizationOutcome.Unavailable;
        }

        if (_candidateId.IsNone)
            _candidateId = _speakerEvidence.NewCandidateId();

        long requiredEnd = checked(wakeEndSample + _postTriggerEvidenceSamples);
        // A detector that reports no keyword extent, with no post-trigger context configured, still
        // has to yield a candidate: the audio available now is its evidence.
        if (requiredEnd <= wakeStartSample)
            requiredEnd = captureEndSample;
        long candidateEnd = Math.Min(captureEndSample, requiredEnd);
        bool inputComplete = captureEndSample >= requiredEnd;
        // A zero-duration KWS estimate carries no focus; windows then anchor on the candidate start.
        long focusEnd = Math.Min(wakeEndSample, captureEndSample);
        bool hasFocus = focusEnd > wakeStartSample;
        var query = new SpeakerEvidenceQuery(
            SpeakerVerificationPurpose.Wakeup,
            timeline.WakeDetectorStreamEpoch,
            timeline.ComputeCandidateStartCaptureSample(wakeStartSample),
            candidateEnd,
            timestampTicks,
            InputComplete: inputComplete,
            FocusStartSample: hasFocus ? wakeStartSample : 0,
            FocusEndSampleExclusive: hasFocus ? focusEnd : 0,
            StrongUpstreamEvidence: IsStrongWakeupEvidence(wakeup),
            CandidateId: _candidateId);
        if (query.CandidateEndSampleExclusive <= query.CandidateStartSample)
            return WakeupAuthorizationOutcome.Waiting;

        SpeakerVerificationResult verification = _speakerEvidence.Evaluate(in query);
        _lastResult = verification;
        switch (verification.Status)
        {
            case SpeakerVerificationStatus.Match:
                return WakeupAuthorizationOutcome.Authorized;
            case SpeakerVerificationStatus.Mismatch:
                return WakeupAuthorizationOutcome.Mismatch;
            case SpeakerVerificationStatus.InsufficientEvidence:
                return WakeupAuthorizationOutcome.InsufficientEvidence;
            case SpeakerVerificationStatus.Unavailable:
            case SpeakerVerificationStatus.Cancelled:
                return WakeupAuthorizationOutcome.Unavailable;
        }

        // Pending: wait for the first complete window; once it exists, the worker gets the bounded
        // inference timeout. The timeout never means "wait for three windows".
        if (verification.Reason != SpeakerVerificationReason.WaitingForInference)
        {
            long evidenceReadyEnd = verification.RequiredEvidenceEndSampleExclusive > 0
                ? Math.Min(verification.RequiredEvidenceEndSampleExclusive, query.CandidateEndSampleExclusive)
                : wakeEndSample;
            if (captureEndSample < evidenceReadyEnd || !inputComplete)
                return WakeupAuthorizationOutcome.Waiting;
        }

        long now = _timeProvider.GetTimestamp();
        if (_pendingReadyTimestamp < 0)
            _pendingReadyTimestamp = now;
        return _timeProvider.GetElapsedTime(_pendingReadyTimestamp, now) <
            _speakerEvidence.GetInferenceTimeout(SpeakerVerificationPurpose.Wakeup)
                ? WakeupAuthorizationOutcome.Waiting
                : WakeupAuthorizationOutcome.Unavailable;
    }

    private bool IsStrongWakeupEvidence(WakeupEventArgs wakeup)
        => (long)wakeup.Score - wakeup.Threshold >= _strongScoreMargin;
}
