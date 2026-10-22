namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishUserSpokeEffect(
    long TurnId,
    string Text,
    TimeSpan SpeechDuration,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
