namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal enum SpeechEndedDecisionKind
{
    None,
    DiscardedBargeInCandidate,
    DiscardedListeningCandidate,
    WaitingForAsrStart,
    ReadyForRecognition,
}
