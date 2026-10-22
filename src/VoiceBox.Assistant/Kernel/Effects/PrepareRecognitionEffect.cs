namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PrepareRecognitionEffect(long TurnId, long OperationId)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.PrepareRecognition;
}
