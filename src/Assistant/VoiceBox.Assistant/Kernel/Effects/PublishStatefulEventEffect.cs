namespace VoiceBox.Assistant.Kernel.Effects;

internal abstract record PublishStatefulEventEffect(long TurnId, long OperationId = 0)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.PublishStatefulEvent;
}
