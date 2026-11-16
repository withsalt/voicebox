using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Gates;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class InputGateEffectHandler : IInputGateEffectHandler
{
    private readonly InputGate _gate;

    public InputGateEffectHandler(InputGate gate)
        => _gate = gate ?? throw new ArgumentNullException(nameof(gate));

    public ValueTask SetInputGateAsync(SetInputGateEffect effect, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (effect.IsOpen)
            _gate.Open();
        else
            _gate.Close();
        return ValueTask.CompletedTask;
    }
}
