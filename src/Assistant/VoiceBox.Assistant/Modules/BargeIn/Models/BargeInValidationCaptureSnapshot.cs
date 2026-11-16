namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct BargeInValidationCaptureSnapshot(
    long TurnId,
    long InterruptedTurnId,
    CancellationTokenSource? ResponseCancellation);
