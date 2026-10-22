using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishWakeModeChangedEffect(
    long TurnId,
    AssistantWakeMode From,
    AssistantWakeMode To,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
