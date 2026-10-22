namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishAssistantResponseGeneratedEffect(
    long TurnId,
    string Text,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
