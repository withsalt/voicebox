namespace VoiceBox.Assistant.Composition.Plans;

internal sealed record AssistantCompositionSnapshot(
    string ProfileId,
    IReadOnlyList<string> Capabilities,
    string CapabilityGraphHash)
{
    public IReadOnlyList<string> CapabilityEdges { get; init; } =
        Array.AsReadOnly(Array.Empty<string>());

    public IReadOnlyList<ModuleOwnership> Ownership { get; init; } =
        Array.AsReadOnly(Array.Empty<ModuleOwnership>());
}
