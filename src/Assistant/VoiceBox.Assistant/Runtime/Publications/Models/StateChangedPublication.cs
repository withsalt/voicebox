using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal sealed record StateChangedPublication(AssistantState From, AssistantState To)
    : AssistantPublication;
