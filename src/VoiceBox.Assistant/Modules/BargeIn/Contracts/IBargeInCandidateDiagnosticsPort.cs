using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Debug-only diagnostics for candidates the boundary discarded. Implementations must not change
/// any turn state.
/// </summary>
internal interface IBargeInCandidateDiagnosticsPort
{
    void LogDiscardedListeningCandidate(SpeechEndedDecision decision);

    void RecordDiscardedPlaybackCandidate(
        SpeechEndedDecision decision,
        AssistantState currentState);
}
