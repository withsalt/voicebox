using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// One confirmation transaction. The lease carries every identity the data plane needs to accept
/// or discard a late commit: operation id, candidate, owner turn, stream epoch and the immutable
/// selection with its owned initial audio.
/// </summary>
internal readonly record struct NearEndConfirmationLease(
    long OperationId,
    SpeakerCandidateId CandidateId,
    long OwnerTurnId,
    long StreamEpoch,
    ActiveNearEndPreRollSelection Selection,
    IReadOnlyList<float[]> InitialAudio,
    BargeInDecision Decision,
    bool EndpointVadCandidate,
    bool IdentityRequired,
    bool IdentityAuthorized,
    SpeakerVerificationResult SpeakerResult);
