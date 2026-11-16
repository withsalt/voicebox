using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Everything the recognition module needs from barge-in while a playback interruption is being
/// validated: telling a validation turn apart, unwinding one whose ASR session failed to start,
/// and releasing candidate context a recognition transition invalidates. It exposes no candidate
/// thresholds, no journal and no validation ownership representation.
/// </summary>
internal interface IBargeInTurnSupportPort
{
    bool IsValidating(long turnId);

    bool TryObserveAsrStartFailure(long turnId, out AssistantState restoredState);

    bool CommitAsrStartFailure(long turnId);

    void ClearValidationPendingAudio();

    void ClearCandidates();

    void CompleteDecisionContext(long turnId);

    void ForgetDecisionContext(long turnId);

    void HandleRejectedStateRestored(AssistantState restoredState);
}
