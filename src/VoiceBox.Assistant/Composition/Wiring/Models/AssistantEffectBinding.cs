using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Composition.Wiring.Models;

internal readonly record struct AssistantEffectBinding(
    AssistantEffectKind Effect,
    AssistantEffectHandlerFamily Handler);
