using System.Runtime.CompilerServices;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>Streams one prompt through the borrowed provider, reusing its stateful conversation.</summary>
internal sealed class TextOnlyTurnExecutor : ITurnExecutor
{
    private readonly IStreamLLMService _service;
    private readonly ILLMConversation? _conversation;
    private int _disposed;

    public TextOnlyTurnExecutor(IStreamLLMService service)
    {
        _service = service ?? throw new ArgumentNullException(nameof(service));
        _conversation = (service as IConversationalLLMService)?.CreateConversation();
    }

    public async IAsyncEnumerable<TurnEvent> ExecuteAsync(
        TurnExecutionContext context,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        LlmCompleteOptions options = context.Options.ToProviderOptions();
        IAsyncEnumerable<LlmTextChunk> stream = _conversation is not null
            ? _conversation.StreamAsync(context.Prompt, options, cancellationToken)
            : _service.StreamAsync(context.Prompt, options, cancellationToken);
        await foreach (LlmTextChunk chunk in stream
            .WithCancellation(cancellationToken)
            .ConfigureAwait(false))
        {
            yield return chunk.Kind == LlmChunkKind.Thinking
                ? new TurnThinkingDelta(chunk.Text)
                : new TurnTextDelta(chunk.Text);
        }

        yield return new TurnFinished();
    }

    public ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return ValueTask.CompletedTask;
        _conversation?.Dispose();
        return ValueTask.CompletedTask;
    }
}
