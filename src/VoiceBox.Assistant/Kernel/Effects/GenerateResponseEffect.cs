namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record GenerateResponseEffect(long TurnId, long OperationId, string Text)
    : AssistantEffect(TurnId, OperationId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.GenerateResponse;
}
