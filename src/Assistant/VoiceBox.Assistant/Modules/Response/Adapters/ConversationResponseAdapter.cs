using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>Protocol-validated update consumed by the response module.</summary>
internal abstract record ConversationTurnEvent;

internal sealed record ConversationTurnThinking(string Text) : ConversationTurnEvent;

internal sealed record ConversationTurnText(string Text, bool EndsTextSegment) : ConversationTurnEvent;

internal sealed record ConversationTurnCompleted(Guid RequestId, string Text) : ConversationTurnEvent;

internal sealed class ConversationProtocolException(string message) : Exception(message);

/// <summary>
/// Associates each Assistant-driven request with its own id, validates the conversation protocol
/// and exposes the foreground output boundary background playback must not overtake.
/// </summary>
internal sealed class ConversationResponseAdapter
{
    private readonly IConversationSession _conversation;
    private readonly int _sampleRate;
    private readonly int _channels;
    private readonly ConcurrentDictionary<Guid, TaskCompletionSource> _pendingForeground = new();

    public ConversationResponseAdapter(IConversationSession conversation, int sampleRate, int channels)
    {
        _conversation = conversation ?? throw new ArgumentNullException(nameof(conversation));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        _sampleRate = sampleRate;
        _channels = channels;
    }

    public string ConversationId => _conversation.ConversationId;

    public async IAsyncEnumerable<ConversationTurnEvent> StreamAsync(
        string text,
        ReadOnlyMemory<float> audio,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        Guid requestId = Guid.NewGuid();
        var boundary = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _pendingForeground[requestId] = boundary;
        bool completed = false;
        var visible = new StringBuilder();
        try
        {
            ConversationAudio? snapshot = audio.IsEmpty ? null : new ConversationAudio(audio.Span, _sampleRate, _channels);
            await foreach (ConversationEvent update in _conversation
                .StreamAsync(new ConversationRequest(requestId, text, snapshot), cancellationToken)
                .WithCancellation(cancellationToken)
                .ConfigureAwait(false))
            {
                switch (update)
                {
                    case ConversationThinkingDelta thinking:
                        if (completed)
                            throw new ConversationProtocolException("The conversation emitted thinking after its completed update.");
                        if (!string.IsNullOrEmpty(thinking.Text))
                            yield return new ConversationTurnThinking(thinking.Text);
                        break;
                    case ConversationTextDelta delta:
                        if (completed)
                            throw new ConversationProtocolException("The conversation emitted text after its completed update.");
                        if (!string.IsNullOrEmpty(delta.Text))
                        {
                            visible.Append(delta.Text);
                            yield return new ConversationTurnText(delta.Text, delta.EndsTextSegment);
                        }
                        break;
                    case ConversationCompleted completion:
                        if (completed)
                            throw new ConversationProtocolException("The conversation emitted more than one completed update.");
                        completed = true;
                        string accumulated = visible.ToString();
                        if (!string.Equals(accumulated, completion.Text, StringComparison.Ordinal))
                        {
                            throw new ConversationProtocolException("The conversation completed update does not match its visible text.");
                        }
                        yield return new ConversationTurnCompleted(requestId, completion.Text);
                        break;
                    default:
                        throw new ConversationProtocolException($"Unsupported conversation update '{update.GetType().Name}'.");
                }
            }

            if (!completed)
            {
                throw new ConversationProtocolException("The conversation ended without a completed update.");
            }
        }
        finally
        {
            // A canceled or failed foreground must not strand background playback.
            boundary.TrySetResult();
            _pendingForeground.TryRemove(requestId, out _);
        }
    }

    /// <summary>
    /// Completes the foreground output boundary for one request. Called after the generation
    /// completion signal has been published, so background results cannot overtake it.
    /// </summary>
    public void ReleaseForegroundBoundary(Guid requestId)
    {
        if (_pendingForeground.TryGetValue(requestId, out TaskCompletionSource? boundary))
            boundary.TrySetResult();
    }

    public async Task WaitForForegroundBoundaryAsync(Guid requestId, CancellationToken cancellationToken)
    {
        if (_pendingForeground.TryGetValue(requestId, out TaskCompletionSource? boundary))
            await boundary.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }
}
