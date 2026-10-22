using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Tracks one playback interruption candidate, including its audio interval, deadline and speaker
/// verdict. The coordinator protects this state with its data-plane lock.
/// </summary>
internal sealed class NearEndSpeakerCandidateState
{
    public bool Active { get; private set; }
    public bool Authorized { get; private set; }
    public bool IdentityRequired { get; private set; }
    public SpeakerCandidateId CandidateId { get; private set; }
    public long OwnerTurnId { get; private set; }
    public BargeInDecision Decision { get; private set; }
    public long StreamEpoch { get; private set; }
    public long MainStartSample { get; private set; } = -1;
    public long EndSample { get; set; }
    public long DeadlineSample { get; private set; }
    public long InferenceReadyTimestamp { get; set; } = -1;
    public bool StrongUpstreamEvidence { get; private set; }
    public bool DtdInterruptObserved { get; private set; }
    public bool EndpointObserved { get; set; }
    public SpeakerVerificationResult SpeakerResult { get; set; }
    public long Generation { get; private set; }

    /// <summary>Opens a KWS-enabled candidate that must wait for its own speaker Match.</summary>
    public void Begin(
        in BargeInDecision decision,
        SpeakerCandidateId candidateId,
        long ownerTurnId,
        long anchorSample,
        long endSampleExclusive,
        long deadlineSample,
        bool strongUpstreamEvidence)
    {
        Active = true;
        Authorized = false;
        IdentityRequired = true;
        CandidateId = candidateId;
        OwnerTurnId = ownerTurnId;
        Decision = decision;
        StreamEpoch = decision.StreamEpoch;
        MainStartSample = anchorSample;
        EndSample = endSampleExclusive;
        DeadlineSample = deadlineSample;
        InferenceReadyTimestamp = -1;
        StrongUpstreamEvidence = strongUpstreamEvidence;
        DtdInterruptObserved = decision.Action == BargeInAction.Interrupt;
        EndpointObserved = false;
        SpeakerResult = default;
        Generation = checked(Generation + 1);
    }

    /// <summary>
    /// Opens a KWS-disabled timeline candidate: the acoustic Interrupt is the whole authorization.
    /// The main recognition stream can lag the detection stream, so the first confirmed decision is
    /// pinned here; a retry must not adopt a continuously advancing window.
    /// </summary>
    public void BeginPreAuthorizedTimeline(in BargeInDecision decision, long ownerTurnId)
    {
        Active = true;
        Authorized = true;
        IdentityRequired = false;
        CandidateId = default;
        OwnerTurnId = ownerTurnId;
        Decision = decision;
        StreamEpoch = decision.StreamEpoch;
        MainStartSample = -1;
        EndSample = decision.EndSampleExclusive;
        DeadlineSample = decision.EndSampleExclusive;
        InferenceReadyTimestamp = -1;
        StrongUpstreamEvidence = false;
        DtdInterruptObserved = true;
        EndpointObserved = false;
        SpeakerResult = default;
        Generation = checked(Generation + 1);
    }

    public void Clear()
    {
        Active = false;
        Authorized = false;
        IdentityRequired = false;
        CandidateId = default;
        OwnerTurnId = 0;
        Decision = default;
        StreamEpoch = 0;
        MainStartSample = -1;
        EndSample = 0;
        DeadlineSample = 0;
        InferenceReadyTimestamp = -1;
        StrongUpstreamEvidence = false;
        DtdInterruptObserved = false;
        EndpointObserved = false;
        SpeakerResult = default;
    }

    public void Authorize() => Authorized = true;

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

    public bool IsCurrent(long generation) => Active && Generation == generation;
}
