using VoiceBox.Assistant.Modules.Response.Models;

namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IResponsePlaybackSession : IDisposable
{
    Task<string> PlayAsync(
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken cancellationToken);
}
