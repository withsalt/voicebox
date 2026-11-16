namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal enum SpeakingDecisionKind
{
    None,
    FeedAsr,
    RecognitionQueueOverflow,
    ConfirmedBargeIn,
    ConfirmedListening,
    RejectedBySpeaker,
}
