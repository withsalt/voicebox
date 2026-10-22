namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct ActiveNearEndProcessingState(
    bool HasActiveCapture,
    bool HasPendingCandidate);
