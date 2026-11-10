using System.Runtime.CompilerServices;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal;

/// <summary>
/// Owns the shared turn protocol for the built-in sessions: serialization, one context retrieval
/// per turn, prompt construction, commit-before-completed and rollback on every non-committed exit.
/// </summary>
internal sealed class ConversationSessionCore : IAsyncDisposable
{
    private readonly SemaphoreSlim _turnGate = new(1, 1);
    private readonly CancellationTokenSource _lifetimeCancellation = new();
    private readonly ConversationHistoryStore _history;
    private readonly LlmConversationOptions _options;
    private readonly IConversationContextProvider[] _contextProviders;
    private readonly ITurnExecutor _executor;
    private readonly object _runLock = new();
    private CancellationTokenSource? _activeRun;
    private long _turnOrdinal;
    private int _disposed;

    public ConversationSessionCore(
        string conversationId,
        ConversationHistoryStore history,
        LlmConversationOptions options,
        IReadOnlyList<IConversationContextProvider> contextProviders,
        ITurnExecutor executor)
    {
        if (string.IsNullOrWhiteSpace(conversationId))
            throw new ArgumentException("Conversation id cannot be empty.", nameof(conversationId));
        ConversationId = conversationId;
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _contextProviders = contextProviders.ToArray();
        _executor = executor ?? throw new ArgumentNullException(nameof(executor));
    }

    public string ConversationId { get; }

    /// <summary>Starts a lifecycle run for sessions that expose Start/Stop.</summary>
    internal void BeginRun()
    {
        lock (_runLock)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            if (_activeRun is not null)
                throw new InvalidOperationException("The conversation session is already running.");
            _activeRun = new CancellationTokenSource();
        }
    }

    /// <summary>
    /// Ends the lifecycle run: cancels an in-flight turn, waits for its rollback, and releases
    /// the run token. This must complete before the owning session stops its providers.
    /// </summary>
    internal async Task EndRunAsync()
    {
        CancellationTokenSource? run;
        lock (_runLock)
        {
            run = _activeRun;
            _activeRun = null;
        }

        if (run is null)
            return;

        try { await run.CancelAsync().ConfigureAwait(false); } catch { }
        await WaitForTurnToFinishAsync().ConfigureAwait(false);
        run.Dispose();
    }

    public IAsyncEnumerable<ConversationEvent> StreamAsync(ConversationRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.Text);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        CancellationToken runToken;
        lock (_runLock)
            runToken = _activeRun?.Token ?? CancellationToken.None;

        return new SingleUseAsyncEnumerable<ConversationEvent>(
            enumeratorToken => StreamCoreAsync(request, cancellationToken, enumeratorToken, runToken));
    }

    private async IAsyncEnumerable<ConversationEvent> StreamCoreAsync(
        ConversationRequest request,
        CancellationToken requestCancellationToken,
        [EnumeratorCancellation] CancellationToken enumeratorCancellationToken,
        CancellationToken runCancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        using var linked = CancellationTokenSource.CreateLinkedTokenSource(
            requestCancellationToken,
            enumeratorCancellationToken,
            runCancellationToken,
            _lifetimeCancellation.Token);
        CancellationToken token = linked.Token;
        await _turnGate.WaitAsync(token).ConfigureAwait(false);

        long turnId = 0;
        bool committed = false;
        TurnCommitGate? commitGate = null;
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            token.ThrowIfCancellationRequested();
            turnId = Interlocked.Increment(ref _turnOrdinal);

            IReadOnlyList<ChatMessage> committedHistory = _history.GetCommittedTranscriptCopy();
            string? additionalContext = await CollectContextAsync(request, committedHistory, token).ConfigureAwait(false);

            _history.AddUserMessage(turnId, request.Text, additionalContext, out ChatMessage[] prompt);
            var executionContext = new TurnExecutionContext(
                turnId,
                request.RequestId,
                ConversationId,
                prompt,
                new ModelGenerationOptions(_options.MaxResponseTokens, _options.EnableThinking));

            var visible = new StringBuilder();
            bool finished = false;
            await foreach (TurnEvent turnEvent in _executor
                .ExecuteAsync(executionContext, token)
                .WithCancellation(token)
                .ConfigureAwait(false))
            {
                switch (turnEvent)
                {
                    case TurnThinkingDelta { Text.Length: > 0 } thinking:
                        yield return new ConversationThinkingDelta(thinking.Text);
                        break;
                    case TurnThinkingDelta:
                        break;
                    case TurnTextDelta { Text.Length: > 0 } text:
                        visible.Append(text.Text);
                        yield return new ConversationTextDelta(text.Text, text.EndsTextSegment);
                        break;
                    case TurnTextDelta:
                        break;
                    case TurnFinished finishedUpdate:
                        if (finished)
                        {
                            throw new InvalidOperationException("The conversation executor completed the turn more than once.");
                        }
                        finished = true;

                        // Final cancellation check: once history commits, a later cancel must not
                        // roll the turn back.
                        token.ThrowIfCancellationRequested();
                        string answer = visible.ToString();
                        if (!_history.CommitGeneratedResponse(turnId, answer))
                        {
                            throw new InvalidOperationException("The conversation turn was not pending when it completed.");
                        }
                        committed = true;
                        commitGate = finishedUpdate.CommitGate;
                        // Background work is released only after the foreground history commit and
                        // before ConversationCompleted becomes observable.
                        finishedUpdate.CommitGate?.Commit();
                        yield return new ConversationCompleted(answer);
                        yield break;
                    default:
                        throw new InvalidOperationException($"Unsupported turn event '{turnEvent.GetType().Name}'.");
                }
            }

            token.ThrowIfCancellationRequested();
            throw new InvalidOperationException("The conversation executor ended without completing the turn.");
        }
        finally
        {
            if (!committed)
            {
                commitGate?.Abort();
                if (turnId > 0)
                    _history.RemoveUnansweredUserMessage(turnId);
            }

            _turnGate.Release();
        }
    }

    private async ValueTask<string?> CollectContextAsync(
        ConversationRequest request,
        IReadOnlyList<ChatMessage> committedHistory,
        CancellationToken cancellationToken)
    {
        if (_contextProviders.Length == 0)
            return null;

        var segments = new List<string>(_contextProviders.Length);
        foreach (IConversationContextProvider provider in _contextProviders)
        {
            string value = await provider.GetContextAsync(
                new ConversationContext(ConversationId, request, committedHistory),
                cancellationToken).ConfigureAwait(false);
            if (!string.IsNullOrWhiteSpace(value))
                segments.Add(value.Trim());
        }

        return segments.Count == 0 ? null : string.Join("\n\n", segments);
    }

    private async Task WaitForTurnToFinishAsync()
    {
        // Wait for the in-flight turn to observe cancellation and release the gate; its finally
        // rolls back uncommitted state. No new turn can pass the disposed check.
        bool entered = false;
        try
        {
            await _turnGate.WaitAsync().ConfigureAwait(false);
            entered = true;
        }
        finally
        {
            if (entered)
                _turnGate.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        CancellationTokenSource? run;
        lock (_runLock)
        {
            run = _activeRun;
            _activeRun = null;
        }

        try { await _lifetimeCancellation.CancelAsync().ConfigureAwait(false); } catch { }
        if (run is not null)
        {
            try { await run.CancelAsync().ConfigureAwait(false); } catch { }
        }

        await WaitForTurnToFinishAsync().ConfigureAwait(false);
        run?.Dispose();

        try { await _executor.DisposeAsync().ConfigureAwait(false); } catch { }
        // The semaphore is deliberately not disposed: a queued request may still be unwinding
        // from its cancellation and could otherwise observe ObjectDisposedException from Release.
        _lifetimeCancellation.Dispose();
    }
}
