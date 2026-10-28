using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Typed, immutable effects produced by near-end state transitions. Data-plane code never calls
/// playback, control ingress or the recognition sink directly: it collects these values under the
/// state lock and hands them to the effect executor after releasing it. PCM is carried only as
/// owned snapshots; spans, providers and mutable services never cross this boundary.
/// </summary>
internal abstract record NearEndEffect
{
    public sealed record DuckPlayback(
        long OwnerTurnId,
        long StreamEpoch,
        float Gain,
        float OriginalGain,
        long Sequence) : NearEndEffect;

    public sealed record RestorePlayback(
        long OwnerTurnId,
        long StreamEpoch,
        long Sequence) : NearEndEffect;

    public sealed record QueueConfirmation(long OperationId) : NearEndEffect;

    public sealed record PublishSpeechEnded(long TurnId, TimeSpan Duration) : NearEndEffect;

    public sealed record CancelSpeakerCandidate(SpeakerCandidateId CandidateId) : NearEndEffect;

    public sealed record ReportDataPlaneFault(AssistantFailure Failure) : NearEndEffect;

    public sealed record ResetRecognition(long TurnId) : NearEndEffect;
}
