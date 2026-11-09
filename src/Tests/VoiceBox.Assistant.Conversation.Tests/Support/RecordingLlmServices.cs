using System.Runtime.CompilerServices;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Tests.Support;

internal sealed class RecordingLlmService : IStreamLLMService, IToolCallingStreamLLMService
{
    private readonly object _lock = new();
    private readonly List<ChatMessage[]> _prompts = [];
    private readonly List<LlmCompleteOptions> _options = [];
    private readonly List<LlmToolTurnRequest> _toolRequests = [];

    public string ResponseText { get; set; } = "好的。";

    public string ThinkingText { get; set; } = string.Empty;

    public IReadOnlyList<LlmTextChunk>? Chunks { get; set; }

    public Func<int, LlmToolTurnRequest, IAsyncEnumerable<LlmTurnEvent>>? ToolTurnScript { get; set; }

    public TaskCompletionSource? BlockAfterFirstChunk { get; set; }

    public int DisposeCount { get; private set; }

    public IReadOnlyList<ChatMessage[]> Prompts
    {
        get { lock (_lock) return _prompts.ToArray(); }
    }

    public IReadOnlyList<LlmCompleteOptions> Options
    {
        get { lock (_lock) return _options.ToArray(); }
    }

    public IReadOnlyList<LlmToolTurnRequest> ToolRequests
    {
        get { lock (_lock) return _toolRequests.ToArray(); }
    }

    public Task<LlmCompletionResult> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        CancellationToken ct = default)
        => Task.FromResult(new LlmCompletionResult { Text = ResponseText });

    public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        int call;
        lock (_lock)
        {
            _prompts.Add(messages.ToArray());
            _options.Add(options ?? new LlmCompleteOptions());
            call = _prompts.Count;
        }

        IReadOnlyList<LlmTextChunk> chunks = Chunks ?? BuildDefaultChunks(options?.EnableThinking == true);
        bool first = true;
        foreach (LlmTextChunk chunk in chunks)
        {
            ct.ThrowIfCancellationRequested();
            yield return chunk;
            if (first && BlockAfterFirstChunk is { } block)
            {
                first = false;
                await block.Task.WaitAsync(ct).ConfigureAwait(false);
            }
            first = false;
        }

        _ = call;
    }

    private IReadOnlyList<LlmTextChunk> BuildDefaultChunks(bool includeThinking)
    {
        var chunks = new List<LlmTextChunk>();
        if (includeThinking && ThinkingText.Length > 0)
            chunks.Add(new LlmTextChunk { Text = ThinkingText, Kind = LlmChunkKind.Thinking });
        if (ResponseText.Length > 0)
            chunks.Add(new LlmTextChunk { Text = ResponseText, Kind = LlmChunkKind.Text });
        chunks.Add(new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true });
        return chunks;
    }

    public async IAsyncEnumerable<LlmTurnEvent> StreamTurnAsync(
        LlmToolTurnRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int call;
        lock (_lock)
        {
            _prompts.Add(request.Messages.ToArray());
            _options.Add(request.Options ?? new LlmCompleteOptions());
            _toolRequests.Add(request);
            call = _prompts.Count;
        }

        if (ToolTurnScript is null)
        {
            if (ResponseText.Length > 0)
            {
                yield return new LlmTurnTextDelta(ResponseText);
            }
            yield return new LlmTurnCompletedEvent();
            yield break;
        }

        await foreach (LlmTurnEvent turnEvent in ToolTurnScript(call, request)
            .WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            yield return turnEvent;
        }
    }

    public void Dispose() => DisposeCount++;
}

internal sealed class ConversationalRecordingLlmService : IConversationalLLMService
{
    public ConversationProbe Conversation { get; } = new();

    public Task<LlmCompletionResult> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        CancellationToken ct = default)
        => Task.FromResult(new LlmCompletionResult { Text = "ok" });

    public IAsyncEnumerable<LlmTextChunk> StreamAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        CancellationToken ct = default)
        => StreamChunks(ct);

    private static async IAsyncEnumerable<LlmTextChunk> StreamChunks([EnumeratorCancellation] CancellationToken ct)
    {
        await Task.Yield();
        ct.ThrowIfCancellationRequested();
        yield return new LlmTextChunk { Text = "ok", Kind = LlmChunkKind.Text };
        yield return new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true };
    }

    public ILLMConversation CreateConversation()
    {
        Conversation.CreatedCount++;
        return Conversation;
    }

    public void Dispose()
    {
    }

    internal sealed class ConversationProbe : ILLMConversation
    {
        public int CreatedCount { get; set; }

        public int DisposeCount { get; private set; }

        public int StreamCount { get; private set; }

        public IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
        {
            StreamCount++;
            return StreamChunks(ct);
        }

        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => Task.FromResult(new LlmCompletionResult { Text = "ok" });

        private static async IAsyncEnumerable<LlmTextChunk> StreamChunks([EnumeratorCancellation] CancellationToken ct)
        {
            await Task.Yield();
            ct.ThrowIfCancellationRequested();
            yield return new LlmTextChunk { Text = "ok", Kind = LlmChunkKind.Text };
            yield return new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true };
        }

        public void Dispose() => DisposeCount++;
    }
}

internal static class ConversationEvents
{
    public static async Task<List<Conversation.Abstractions.ConversationEvent>> CollectAsync(
        IAsyncEnumerable<Conversation.Abstractions.ConversationEvent> stream,
        CancellationToken cancellationToken = default)
    {
        var updates = new List<Conversation.Abstractions.ConversationEvent>();
        await foreach (Conversation.Abstractions.ConversationEvent update in stream
            .WithCancellation(cancellationToken))
        {
            updates.Add(update);
        }
        return updates;
    }
}
