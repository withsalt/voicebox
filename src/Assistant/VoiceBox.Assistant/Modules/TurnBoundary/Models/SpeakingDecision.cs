using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal readonly record struct SpeakingDecision(
    SpeakingDecisionKind Kind,
    string? AsrSessionId,
    CancellationTokenSource? ResponseToCancel,
    long InterruptedTurnId,
    AssistantState OldState,
    long TurnId,
    long ConfirmedCandidateSamples,
    double ConfirmedCandidateRms,
    long ConfirmedConfirmationSamples,
    double ConfirmedConfirmationRms,
    int ConfirmationMinDurationMs,
    float ConfirmationMinRms,
    int ConfirmationInitialSuppressionMs,
    BargeInDecisionDiagnostics ConfirmedBargeInDiagnostics);
