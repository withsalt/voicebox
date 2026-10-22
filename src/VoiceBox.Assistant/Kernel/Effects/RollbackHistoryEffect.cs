namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record RollbackHistoryEffect(long TurnId, long OperationId = 0)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.RollbackHistory;
}
