using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Confirmation sent to the control loop. With KWS enabled, only this candidate's speaker Match
/// authorizes it; otherwise an acoustic DTD Interrupt is required.
/// </summary>
internal readonly record struct ActiveNearEndConfirmationRequest(
    long OperationId,
    IReadOnlyList<float[]> InitialAudio,
    BargeInDecision Decision,
    ActiveNearEndPreRollSelection Selection,
    bool EndpointVadCandidate,
    bool IdentityRequired,
    bool IdentityAuthorized,
    SpeakerCandidateId CandidateId,
    long OwnerTurnId,
    SpeakerVerificationResult SpeakerResult);
