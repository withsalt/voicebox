using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal readonly record struct SpeechStartedDecision(
    SpeechStartedDecisionKind Kind,
    AssistantState OldState,
    AssistantState CurrentState,
    long TurnId);
