namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record CommitHistoryEffect(long TurnId, string Text, long OperationId = 0)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.CommitHistory;
}
