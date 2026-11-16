using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal sealed record AssistantResponseGeneratedPublication(AssistantResponseGeneratedEventArgs Args)
    : AssistantPublication;
