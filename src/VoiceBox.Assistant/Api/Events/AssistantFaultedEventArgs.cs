namespace VoiceBox.Assistant.Api.Events;

/// <summary>
/// Reports an unrecoverable runtime component failure. The Assistant stops after raising
/// this event and does not switch to another implementation.
/// </summary>
public sealed class AssistantFaultedEventArgs : EventArgs
{
    public required string Component { get; init; }
    public required Exception Exception { get; init; }
}
