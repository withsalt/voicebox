namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// Requests speaker evidence for one exact candidate interval on the identity capture timeline.
/// Strong upstream evidence may reduce the vote count, but never bypasses a current speaker match.
/// </summary>
internal readonly record struct SpeakerEvidenceQuery(
    SpeakerVerificationPurpose Purpose,
    long StreamEpoch,
    long CandidateStartSample,
    long CandidateEndSampleExclusive,
    long QueryTimestampTicks,
    bool InputComplete = false,
    long FocusStartSample = 0,
    long FocusEndSampleExclusive = 0,
    bool StrongUpstreamEvidence = false,
    SpeakerCandidateId CandidateId = default,
    long OwnerTurnId = 0)
{
    /// <summary>
    /// Exact speech interval supplied by a detector such as KWS. When present, wakeup evidence
    /// windows are centered around this interval instead of the trailing capture range.
    /// </summary>
    public bool HasFocusInterval => FocusEndSampleExclusive > FocusStartSample;
}
