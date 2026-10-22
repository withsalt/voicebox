using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Composition.Wiring;

internal sealed class ModuleWiring
{
    private static readonly AssistantEffectBinding[] DefaultEffectBindings =
    [
        new(AssistantEffectKind.SetInputGate, AssistantEffectHandlerFamily.InputGate),
        new(AssistantEffectKind.PrepareRecognition, AssistantEffectHandlerFamily.Recognition),
        new(AssistantEffectKind.FinishRecognition, AssistantEffectHandlerFamily.Recognition),
        new(AssistantEffectKind.GenerateResponse, AssistantEffectHandlerFamily.ResponseGeneration),
        new(AssistantEffectKind.CancelResponse, AssistantEffectHandlerFamily.ResponseGeneration),
        new(AssistantEffectKind.ClearPlaybackQueue, AssistantEffectHandlerFamily.Playback),
        new(AssistantEffectKind.StopPlayback, AssistantEffectHandlerFamily.Playback),
        new(AssistantEffectKind.CommitHistory, AssistantEffectHandlerFamily.Conversation),
        new(AssistantEffectKind.RollbackHistory, AssistantEffectHandlerFamily.Conversation),
        new(AssistantEffectKind.PublishStatefulEvent, AssistantEffectHandlerFamily.PublicEventPublisher),
        new(AssistantEffectKind.CancelOperation, AssistantEffectHandlerFamily.Runtime),
        new(AssistantEffectKind.CompleteTurn, AssistantEffectHandlerFamily.Runtime),
    ];

    public ModuleWiring(AssistantRuntimePlan runtime, IReadOnlyList<AssistantEffectBinding> effectBindings)
    {
        Runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        EffectBindings = Array.AsReadOnly(effectBindings.ToArray());
        ValidateEffectBindings(effectBindings);
    }

    public AssistantRuntimePlan Runtime { get; }

    public IReadOnlyList<AssistantEffectBinding> EffectBindings { get; }

    public static ModuleWiring CreateDefault(AssistantRuntimePlan runtime)
        => new(runtime, DefaultEffectBindings);

    private static void ValidateEffectBindings(IReadOnlyList<AssistantEffectBinding> bindings)
    {
        foreach (AssistantEffectKind effect in Enum.GetValues<AssistantEffectKind>())
        {
            int matchingCount = bindings.Count(binding => binding.Effect == effect);
            if (matchingCount != 1)
            {
                throw new AssistantCompositionException(
                    "VBX-COMP-007",
                    $"Effect '{effect}' must have exactly one typed handler binding; found {matchingCount}.");
            }
        }
    }
}
