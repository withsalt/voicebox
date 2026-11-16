using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Api.Events;

public sealed class AssistantStateChangedEventArgs : EventArgs
{
    public required AssistantState From { get; init; }
    public required AssistantState To { get; init; }
}
