using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Narrow owner port for the mutable VAD-gate verification-candidate session. The gate policy may
/// ask for a query or resolve a version, but it does not retain candidate state itself.
/// </summary>
internal interface ISpeakerVerificationStatePort
{
    bool IsCandidateActive { get; }

    long CurrentVersion { get; }

    void BeginCandidate(SpeakerVerificationPurpose purpose);

    void Reset();

    bool TryBuildQuery(
        long version,
        long streamEpoch,
        long startSample,
        long endSample,
        long queryTimestampTicks,
        out SpeakerEvidenceQuery query);

    bool TryCompleteQuery(long version, long queryTimestampTicks, out SpeakerEvidenceQuery query);

    bool IsCurrent(long version);

    bool TryResolve(long version);
}
