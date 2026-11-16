using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Api.Events;

public sealed class AssistantWakeupDetectedEventArgs : EventArgs
{
    public required WakeupEventArgs Wakeup { get; init; }
}
