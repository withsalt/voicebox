using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Contracts;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>
/// Publishes completed background results through the normal text surface and an owned response
/// turn, so playback participates in the same interruption lifecycle as foreground replies.
/// </summary>
internal sealed class PlaybackBackgroundToolResultSink(
    IBackgroundResponsePlaybackPort playback,
    IAssistantTextStreamSink textStream,
    IConversationBackgroundTextPort history) : IBackgroundToolResultSink
{
    private readonly IBackgroundResponsePlaybackPort _playback = playback
        ?? throw new ArgumentNullException(nameof(playback));
    private readonly IAssistantTextStreamSink _textStream = textStream
        ?? throw new ArgumentNullException(nameof(textStream));
    private readonly IConversationBackgroundTextPort _history = history
        ?? throw new ArgumentNullException(nameof(history));

    public async Task PublishAsync(
        long turnId,
        string text,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;

        _history.RecordBackgroundAssistantText(turnId, text);
        _textStream.RaiseAssistantText(new AssistantTextEventArgs
        {
            Delta = text,
            IsThinking = false,
        });
        await _playback.PlayAsync(text, cancellationToken).ConfigureAwait(false);
    }
}
