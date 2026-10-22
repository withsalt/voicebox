namespace VoiceBox.Assistant.Composition.Plans;

internal readonly record struct CapabilityEdge(
    string ModuleId,
    AssistantCapability Capability,
    CapabilityEdgeKind Kind,
    int Order = 0);
