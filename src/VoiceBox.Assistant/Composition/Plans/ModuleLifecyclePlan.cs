namespace VoiceBox.Assistant.Composition.Plans;

internal sealed class ModuleLifecyclePlan
{
    public ModuleLifecyclePlan(IReadOnlyList<AssistantLifecycleParticipant> startOrder)
    {
        StartOrder = Array.AsReadOnly(startOrder.ToArray());
    }

    public IReadOnlyList<AssistantLifecycleParticipant> StartOrder { get; }
}
