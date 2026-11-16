namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record CancelOperationEffect(long TurnId, long OperationId)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.CancelOperation;
}
