using VoiceBox.Assistant.Modules.Playback.Models;

namespace VoiceBox.Assistant.Modules.Playback.Contracts;

/// <summary>Executes one admitted logical playback request.</summary>
internal interface ISpeechPlaybackRequestExecutor
{
    Task<string> ExecuteAsync(
        SpeechPlaybackRequest request,
        CancellationToken cancellationToken);
}
