using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Tracks the playback-interruption candidate: its lifecycle (<see cref="Status"/>), immutable
/// window, deadline and speaker verdict. Callers may only move it through these methods; no flat
/// boolean fields are exposed. All mutation happens under the near-end data-plane lock.
/// </summary>
internal sealed class NearEndCandidateStateMachine
{
    private long _pendingWindowStart = -1;
    private long _pendingWindowEnd = -1;

    public enum Status
    {
        None,
        AwaitingSpeaker,
        Authorized,
    }

    public Status State { get; private set; }
    public bool Active => State != Status.None;
    public bool Authorized => State == Status.Authorized;
    public bool IdentityRequired { get; private set; }
    public SpeakerCandidateId CandidateId { get; private set; }
    public long OwnerTurnId { get; private set; }
    public BargeInDecision Decision { get; private set; }
    public long StreamEpoch { get; private set; }
    public long MainStartSample { get; private set; } = -1;
    public long EndSample { get; private set; }
    public long DeadlineSample { get; private set; }
    public long InferenceReadyTimestamp { get; private set; } = -1;
    public bool StrongUpstreamEvidence { get; private set; }
    public bool DtdInterruptObserved { get; private set; }
    public bool EndpointObserved { get; private set; }
    public SpeakerVerificationResult SpeakerResult { get; private set; }
    public long Generation { get; private set; }

    /// <summary>Opens an identity-confirmed candidate that must wait for its own speaker Match.</summary>
    public void BeginSpeaker(
        in BargeInDecision decision,
        SpeakerCandidateId candidateId,
        long ownerTurnId,
        long anchorSample,
        long endSampleExclusive,
        long deadlineSample,
        bool strongUpstreamEvidence)
    {
        State = Status.AwaitingSpeaker;
        IdentityRequired = true;
        CandidateId = candidateId;
        OwnerTurnId = ownerTurnId;
        Decision = decision;
        StreamEpoch = decision.StreamEpoch;
        MainStartSample = anchorSample;
        EndSample = endSampleExclusive;
        DeadlineSample = deadlineSample;
        InferenceReadyTimestamp = -1;
        _pendingWindowStart = _pendingWindowEnd = -1;
        StrongUpstreamEvidence = strongUpstreamEvidence;
        DtdInterruptObserved = decision.Action == BargeInAction.Interrupt;
        EndpointObserved = false;
        SpeakerResult = default;
        Generation = checked(Generation + 1);
    }

    /// <summary>
    /// Opens an acoustic timeline candidate: the Interrupt is the whole authorization. The main
    /// recognition stream can lag the detection stream, so the first confirmed decision is pinned
    /// here; a retry must not adopt a continuously advancing window.
    /// </summary>
    public void BeginAuthorizedTimeline(in BargeInDecision decision, long ownerTurnId)
    {
        State = Status.Authorized;
        IdentityRequired = false;
        CandidateId = default;
        OwnerTurnId = ownerTurnId;
        Decision = decision;
        StreamEpoch = decision.StreamEpoch;
        MainStartSample = -1;
        EndSample = decision.EndSampleExclusive;
        DeadlineSample = decision.EndSampleExclusive;
        InferenceReadyTimestamp = -1;
        _pendingWindowStart = _pendingWindowEnd = -1;
        StrongUpstreamEvidence = false;
        DtdInterruptObserved = true;
        EndpointObserved = false;
        SpeakerResult = default;
        Generation = checked(Generation + 1);
    }

    public void Clear()
    {
        State = Status.None;
        IdentityRequired = false;
        CandidateId = default;
        OwnerTurnId = 0;
        Decision = default;
        StreamEpoch = 0;
        MainStartSample = -1;
        EndSample = 0;
        DeadlineSample = 0;
        InferenceReadyTimestamp = -1;
        _pendingWindowStart = _pendingWindowEnd = -1;
        StrongUpstreamEvidence = false;
        DtdInterruptObserved = false;
        EndpointObserved = false;
        SpeakerResult = default;
    }

    public void Authorize()
    {
        if (State == Status.AwaitingSpeaker)
            State = Status.Authorized;
    }

    /// <summary>
    /// Records the DTD's newest view of the same candidate window. Interrupt only ever adds
    /// acoustic evidence (and the strong-upstream flag); it can never authorize on its own.
    /// </summary>
    public void ObserveDecision(in BargeInDecision decision, bool strongNearEnd)
    {
        if (!Active || decision.StreamEpoch != StreamEpoch)
            return;
        Decision = decision;
        if (decision.Action == BargeInAction.Interrupt)
        {
            DtdInterruptObserved = true;
            StrongUpstreamEvidence |= strongNearEnd;
        }
    }

    /// <summary>Extends the evidence window while the endpoint has not frozen it.</summary>
    public void ExtendEndSample(long endSampleExclusive)
        => EndSample = Math.Max(EndSample, endSampleExclusive);

    /// <summary>
    /// Freezes the candidate interval at the endpoint: trailing silence never displaces the
    /// utterance's own windows from the evidence, and the deadline still bounds the window.
    /// </summary>
    public void MarkEndpointObserved(long endSampleExclusive)
    {
        EndpointObserved = true;
        EndSample = Math.Min(Math.Max(EndSample, endSampleExclusive), DeadlineSample);
    }

    /// <summary>
    /// Each outstanding worker window gets its own deadline. Waiting for more audio ends the
    /// previous wait; repeated polls of the same job must never extend it.
    /// </summary>
    public void ObserveInferenceWait(in SpeakerVerificationResult result, long timestamp)
    {
        if (result.Status != SpeakerVerificationStatus.Pending ||
            result.Reason != SpeakerVerificationReason.WaitingForInference)
        {
            InferenceReadyTimestamp = -1;
            _pendingWindowStart = _pendingWindowEnd = -1;
            return;
        }

        bool hasWindow = result.PendingWindowStartSample >= 0 &&
            result.PendingWindowEndSampleExclusive > result.PendingWindowStartSample;
        if (InferenceReadyTimestamp < 0 ||
            (hasWindow && (_pendingWindowStart != result.PendingWindowStartSample ||
                           _pendingWindowEnd != result.PendingWindowEndSampleExclusive)))
        {
            InferenceReadyTimestamp = timestamp;
        }
        if (hasWindow)
        {
            _pendingWindowStart = result.PendingWindowStartSample;
            _pendingWindowEnd = result.PendingWindowEndSampleExclusive;
        }
    }

    public void SetSpeakerResult(in SpeakerVerificationResult result)
    {
        if (Active)
            SpeakerResult = result;
    }

    public bool IsCurrent(long generation) => Active && Generation == generation;
}
