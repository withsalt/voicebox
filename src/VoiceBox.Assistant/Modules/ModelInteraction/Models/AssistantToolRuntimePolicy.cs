namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

/// <summary>Host-owned policy the tool runtime enforces, resolved from frozen assistant options.</summary>
internal readonly record struct AssistantToolRuntimePolicy(bool ValidateArguments);
