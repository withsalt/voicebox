namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishAssistantRepliedEffect(
    long TurnId,
    string Text,
    TimeSpan Elapsed,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
