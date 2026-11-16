namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record FinishRecognitionEffect(long TurnId, long OperationId, string SessionId)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.FinishRecognition;
}
