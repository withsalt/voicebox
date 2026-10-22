namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record StopPlaybackEffect(long TurnId, long OperationId = 0)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.StopPlayback;
}
