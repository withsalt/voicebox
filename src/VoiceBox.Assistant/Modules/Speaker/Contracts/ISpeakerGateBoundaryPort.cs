using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Speaker-gate surface used by the turn boundary across one candidate: begin, evaluate the latest
/// capture audio, and complete on the endpoint. The gate's own versioning, worker scheduling and
/// pending-query state stay inside the coordinator. When no online identity purpose is effective
/// (Confirmation.Enable off) the gate is a pass-through and never touches the provider.
/// </summary>
internal interface ISpeakerGateBoundaryPort : ISpeakerGateCapturePort
{
    /// <summary>Whether this gate verifies identity at all; false means <c>NotRequired</c>.</summary>
    bool IsIdentityRequired { get; }

    /// <summary>Whether the optional single-talk purpose is effective for this composition.</summary>
    bool IsSingleTalkEffective { get; }

    /// <summary>The purpose of the active candidate, or null when no candidate is active.</summary>
    SpeakerVerificationPurpose? ActivePurpose { get; }

    void BeginCandidate(SpeakerVerificationPurpose purpose);

    BargeInSpeakerGateStatus Evaluate(ReadOnlySpan<float> latestAudio);

    SpeechBoundaryObservation CompleteInput(TimeSpan duration);

    /// <summary>The typed reason of the last denied verdict, for diagnostics and resolution reasons.</summary>
    SpeakerVerificationStatus LastDeniedStatus { get; }
}
