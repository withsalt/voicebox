namespace VoiceBox.Assistant.Api.Events;

public sealed class UserSpokeEventArgs : EventArgs
{
    public required string Text { get; init; }
    public required TimeSpan SpeechDuration { get; init; }
}
