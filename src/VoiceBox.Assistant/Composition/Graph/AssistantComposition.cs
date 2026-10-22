using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Composition.Wiring;

namespace VoiceBox.Assistant.Composition.Graph;

internal sealed class AssistantComposition
{
    public AssistantComposition(
        AssistantRuntimePlan runtimePlan,
        ModuleWiring wiring,
        ModuleLifecyclePlan lifecycle,
        AssistantCompositionSnapshot snapshot)
    {
        RuntimePlan = runtimePlan ?? throw new ArgumentNullException(nameof(runtimePlan));
        Wiring = wiring ?? throw new ArgumentNullException(nameof(wiring));
        Lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
        Snapshot = snapshot ?? throw new ArgumentNullException(nameof(snapshot));
    }

    public AssistantRuntimePlan RuntimePlan { get; }

    public ModuleWiring Wiring { get; }

    public ModuleLifecyclePlan Lifecycle { get; }

    public AssistantCompositionSnapshot Snapshot { get; }
}
