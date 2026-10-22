using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct AssistantWakeupObservation(
    AssistantWakeMode FromMode,
    AssistantWakeMode ToMode);
