namespace VoiceBox.Assistant.Composition.Plans;

internal readonly record struct ModuleOwnership(
    string ModuleId,
    ResourceOwnership Ownership);
