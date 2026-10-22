namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal enum SpeechStartedDecisionKind
{
    StartListening,
    ListeningCandidate,
    BargeInCandidate,
    Ignored,
}
