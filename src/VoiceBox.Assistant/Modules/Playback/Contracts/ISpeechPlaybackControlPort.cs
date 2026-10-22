using VoiceBox.Assistant.Api.Playback;

namespace VoiceBox.Assistant.Modules.Playback.Contracts;

/// <summary>Host-facing command port for logical speech playback.</summary>
internal interface ISpeechPlaybackControlPort
{
    ValueTask<SpeechEnqueueResult> EnqueueSpeechAsync(
        string text,
        SpeechPlaybackQueuePosition queuePosition,
        CancellationToken cancellationToken);

    void StopCurrentSpeechPlayback();
}
