namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record SetInputGateEffect(long TurnId, bool IsOpen)
    : AssistantEffect(TurnId)
{
    public override AssistantEffectKind Kind => AssistantEffectKind.SetInputGate;
}
