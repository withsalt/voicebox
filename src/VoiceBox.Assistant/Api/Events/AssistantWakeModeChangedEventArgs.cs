namespace VoiceBox.Assistant.Api.Events;

public sealed class AssistantWakeModeChangedEventArgs : EventArgs
{
    public required AssistantWakeMode From { get; init; }
    public required AssistantWakeMode To { get; init; }
}
