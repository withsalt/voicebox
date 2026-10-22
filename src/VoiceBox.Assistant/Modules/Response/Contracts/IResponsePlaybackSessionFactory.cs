using VoiceBox.Assistant.Modules.Response.Models;

namespace VoiceBox.Assistant.Modules.Response.Contracts;

internal interface IResponsePlaybackSessionFactory
{
    IResponsePlaybackSession Create(long turnId);

    async Task<string> PlayAsync(
        long turnId,
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken cancellationToken)
    {
        using IResponsePlaybackSession session = Create(turnId);
        return await session.PlayAsync(text, cancellationToken).ConfigureAwait(false);
    }
}
