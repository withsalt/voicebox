using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

// Typed capability-family effect ports. Implementations are supplied by the Runtime shell at
// construction time; the composition plan never performs service lookup or dynamic dispatch.
internal interface IInputGateEffectHandler
{
    ValueTask SetInputGateAsync(SetInputGateEffect effect, CancellationToken cancellationToken);
}
