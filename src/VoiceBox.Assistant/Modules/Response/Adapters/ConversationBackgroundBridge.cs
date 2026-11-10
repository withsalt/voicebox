using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Contracts;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>
/// Consumes results the conversation produces after a request's foreground stream ended and hands
/// each text to the existing public text surface and background playback scheduler. It never writes
/// history or calls the model itself.
/// </summary>
internal sealed class ConversationBackgroundBridge(
    IConversationBackgroundOutput output,
    ConversationResponseAdapter conversation,
    IBackgroundResponsePlaybackPort playback,
    IAssistantTextStreamSink textStream,
    ILogger logger)
{
    private readonly IConversationBackgroundOutput _output = output ?? throw new ArgumentNullException(nameof(output));
    private readonly ConversationResponseAdapter _conversation = conversation ?? throw new ArgumentNullException(nameof(conversation));
    private readonly IBackgroundResponsePlaybackPort _playback = playback ?? throw new ArgumentNullException(nameof(playback));
    private readonly IAssistantTextStreamSink _textStream = textStream ?? throw new ArgumentNullException(nameof(textStream));
    private readonly ILogger _logger = logger ?? throw new ArgumentNullException(nameof(logger));

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        try
        {
            await foreach (ConversationBackgroundMessage message in _output
                .ReadBackgroundMessagesAsync(cancellationToken)
                .WithCancellation(cancellationToken)
                .ConfigureAwait(false))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (string.IsNullOrWhiteSpace(message.Text))
                    continue;
                if (!string.Equals(message.ConversationId, _conversation.ConversationId, StringComparison.Ordinal))
                {
                    _logger.LogWarning(
                        "Dropping a background conversation message from another session: {ConversationId}.",
                        message.ConversationId);
                    continue;
                }

                try
                {
                    await _conversation.WaitForForegroundBoundaryAsync(
                        message.OriginatingRequestId,
                        cancellationToken).ConfigureAwait(false);
                    _textStream.RaiseAssistantText(new AssistantTextEventArgs
                    {
                        Delta = message.Text,
                        IsThinking = false,
                    });
                    await _playback.PlayAsync(message.Text, cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exception)
                {
                    _logger.LogError(
                        exception,
                        "Background conversation playback failed for request {RequestId}.",
                        message.OriginatingRequestId);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
    }
}
