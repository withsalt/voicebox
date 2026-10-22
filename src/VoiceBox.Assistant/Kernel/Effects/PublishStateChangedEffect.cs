using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishStateChangedEffect(
    long TurnId,
    AssistantState From,
    AssistantState To,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
