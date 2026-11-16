namespace VoiceBox.Assistant.Api.Playback;

/// <summary>Specifies where a speech request is inserted in the pending playback queue.</summary>
public enum SpeechPlaybackQueuePosition
{
    /// <summary>Append after all requests that are already waiting.</summary>
    Tail = 0,

    /// <summary>
    /// Insert before all requests that are already waiting. The request currently playing is not
    /// interrupted; use <see cref="VoiceAssistant.StopCurrentSpeechPlayback"/> to stop it.
    /// </summary>
    Head = 1,
}
