using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal readonly record struct SpeechEndedDecision(
    SpeechEndedDecisionKind Kind,
    string? AsrSessionId,
    long TurnId,
    TimeSpan SpeechDuration,
    long DiscardedCandidateSamples,
    double DiscardedCandidateRms,
    long DiscardedConfirmationSamples,
    double DiscardedConfirmationRms,
    BargeInDecisionDiagnostics DiscardedBargeInDiagnostics);
