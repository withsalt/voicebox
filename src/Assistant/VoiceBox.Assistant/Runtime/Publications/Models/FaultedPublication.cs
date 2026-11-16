using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal sealed record FaultedPublication(AssistantFaultedEventArgs Args)
    : AssistantPublication;
