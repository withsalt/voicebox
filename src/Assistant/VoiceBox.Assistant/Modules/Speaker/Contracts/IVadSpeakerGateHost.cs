using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Runtime-side collaborator that admits a speaker-gate decision to the control plane. The
/// coordinator owns the gate's own versioning and query state; the shell owns admission.
/// </summary>
internal interface IVadSpeakerGateHost
{
    CancellationToken RunToken { get; }

    bool IsRunCancelled { get; }

    long NextOperationId();

    void SubmitGateResolved(
        long operationId,
        SpeakerVerificationPurpose purpose,
        BargeInSpeakerGateStatus status,
        TimeSpan? pendingSpeechEnd);

    /// <summary>
    /// Completes the boundary directly when the composition has no speaker evidence, instead of
    /// waiting for a verdict that will never arrive.
    /// </summary>
    SpeechBoundaryObservation SpeechEndedWithoutSpeakerGate(TimeSpan duration);

    void Track(string taskName, Task task);
}
