using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

internal readonly record struct WakeupInterruptDecision(
    AssistantState OldState,
    long TurnId,
    long InterruptedTurnId,
    CancellationTokenSource? ResponseToCancel,
    string? AsrSessionIdToCleanup);
