using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Models;

internal readonly record struct RecognitionOutcome(
    long TurnId,
    long OperationId,
    RecognitionOutcomeKind Kind,
    string? RawText,
    string Text,
    TimeSpan SpeechDuration,
    AssistantFailure Failure);
