namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct AssistantFailure(
    string Code,
    string Category,
    string Component,
    string MessageKey);
