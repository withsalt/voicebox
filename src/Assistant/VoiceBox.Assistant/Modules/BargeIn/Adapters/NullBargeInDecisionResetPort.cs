using VoiceBox.Assistant.Modules.BargeIn.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

internal sealed class NullBargeInDecisionResetPort : IBargeInDecisionResetPort
{
    public static NullBargeInDecisionResetPort Instance { get; } = new();

    private NullBargeInDecisionResetPort() { }

    public void ResetDecisionState() { }
}
