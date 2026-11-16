namespace VoiceBox.Assistant.Kernel.Effects;

internal abstract record AssistantEffect(long TurnId, long OperationId = 0)
{
    public abstract AssistantEffectKind Kind { get; }
}
