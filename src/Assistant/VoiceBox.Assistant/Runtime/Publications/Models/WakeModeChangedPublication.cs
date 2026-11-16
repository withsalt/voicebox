using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal sealed record WakeModeChangedPublication(AssistantWakeMode From, AssistantWakeMode To)
    : AssistantPublication;
