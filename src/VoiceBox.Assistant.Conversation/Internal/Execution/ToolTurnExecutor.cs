using System.Diagnostics;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>
/// Executes a bounded native tool loop and keeps provider transcript, user-visible text and tool
/// execution separate. Background calls publish a fixed trusted acknowledgement immediately.
/// </summary>
internal sealed class ToolTurnExecutor : ITurnExecutor
{
    private const int MaxModelRounds = 8;
    private const int MaxToolCallsPerTurn = 16;

    private readonly StructuredLlmTurnClient _client;
    private readonly ConversationToolRuntime _runtime;
    private readonly ConversationHistoryStore _history;
    private readonly IBackgroundResultPublisher _backgroundOutput;
    private readonly ConversationToolOptions _options;
    private readonly object _taskGate = new();
    private readonly HashSet<Task> _backgroundTasks = [];
    private Task _backgroundPublicationTail = Task.CompletedTask;
    private CancellationTokenSource? _runCancellation;
    private ToolCatalogSnapshot? _catalog;
    private int _disposed;

    public ToolTurnExecutor(
        StructuredLlmTurnClient client,
        ConversationToolRuntime runtime,
        ConversationHistoryStore history,
        IBackgroundResultPublisher backgroundOutput,
        ConversationToolOptions options)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _backgroundOutput = backgroundOutput ?? throw new ArgumentNullException(nameof(backgroundOutput));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_runCancellation is not null)
            throw new InvalidOperationException("The tool conversation session is already started.");

        await _runtime.StartAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ToolCatalogSnapshot catalog = await _runtime.GetSnapshotAsync(cancellationToken)
                .ConfigureAwait(false);
            if (catalog.IsEmpty)
                throw new InvalidOperationException("The tool conversation session exposed no tools.");
            _catalog = catalog;
            _runCancellation = new CancellationTokenSource();
        }
        catch
        {
            await _runtime.StopAsync().ConfigureAwait(false);
            throw;
        }
    }

    public async Task StopAsync()
    {
        CancellationTokenSource? run = Interlocked.Exchange(ref _runCancellation, null);
        if (run is not null)
        {
            try { await run.CancelAsync().ConfigureAwait(false); } catch { }
        }

        Task[] tasks;
        lock (_taskGate)
            tasks = _backgroundTasks.ToArray();
        try { await Task.WhenAll(tasks).ConfigureAwait(false); } catch { }
        try { await _runtime.StopAsync().ConfigureAwait(false); } finally { run?.Dispose(); }
        _catalog = null;
    }

    public async IAsyncEnumerable<TurnEvent> ExecuteAsync(
        TurnExecutionContext context,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ToolCatalogSnapshot catalog = _catalog
            ?? throw new InvalidOperationException("The tool conversation session is not started.");

        TurnCommitGate? pendingGate = null;
        var transcript = context.Prompt.ToList();
        var retryBlockedAfterUnknownOutcome = new UnknownOutcomeBlocklist();
        // Provider transcript slice this turn produced, reported so a later turn can still observe
        // structured tool results instead of only the spoken answer.
        var toolTranscript = new List<ChatMessage>();
        int totalToolCalls = 0;
        try
        {
            for (int round = 0; round < MaxModelRounds; round++)
            {
                var calls = new List<ModelTurnToolCall>();
                var visibleText = new List<VisibleTextDelta>();
                await foreach (ModelTurnEvent modelEvent in _client.RunAsync(
                    new ModelTurnContext(
                        transcript,
                        context.Options,
                        catalog,
                        context.TurnId,
                        context.RequestId),
                    cancellationToken).ConfigureAwait(false))
                {
                    switch (modelEvent)
                    {
                        case ThinkingDelta thinking:
                            yield return new TurnThinkingDelta(thinking.Text);
                            break;
                        case ModelTurnToolCall call:
                            calls.Add(call);
                            break;
                        case VisibleTextDelta { Text.Length: > 0 } delta:
                            visibleText.Add(delta);
                            break;
                        case VisibleTextDelta:
                        case ModelTurnCompleted:
                            break;
                        default:
                            throw new InvalidOperationException(
                                $"Unsupported model turn event '{modelEvent.GetType().Name}'.");
                    }
                }

                // Providers may emit assistant text alongside native tool calls. Keep that
                // intermediate narration out of speech and history; only a round without tool
                // calls supplies the final visible answer.
                if (calls.Count == 0)
                {
                    PublishToolTranscript(context.TurnId, toolTranscript);
                    foreach (VisibleTextDelta delta in visibleText)
                        yield return new TurnTextDelta(delta.Text);
                    yield return new TurnFinished();
                    yield break;
                }

                totalToolCalls += calls.Count;
                if (totalToolCalls > MaxToolCallsPerTurn)
                    throw new InvalidOperationException("The model exceeded the per-turn tool call limit.");

                ToolCall[] resolved = calls.Select(call => ResolveCall(call, catalog)).ToArray();
                // A provider tool-call batch must be resolved atomically. If it mixes background and
                // foreground preferences, foreground wins so every tool result can be returned in
                // the same provider transcript without leaving dangling call ids.
                bool runInBackground = resolved.All(static call =>
                    call.ExecutionPreference == ConversationToolExecutionPreference.Background);
                string? executionPrompt = FindCustomBackgroundExecutionPrompt(resolved, catalog);
                ChatMessage callMessage = CreateAssistantToolCallMessage(resolved, catalog);
                transcript.Add(callMessage);

                if (runInBackground)
                {
                    var commitGate = new TurnCommitGate();
                    pendingGate = commitGate;
                    ScheduleBackground(
                        context,
                        resolved,
                        transcript.ToArray(),
                        catalog,
                        retryBlockedAfterUnknownOutcome,
                        commitGate,
                        toolTranscript);
                    try
                    {
                        yield return new TurnTextDelta(
                            executionPrompt ?? CreateStandardBackgroundExecutionPrompt(resolved, catalog),
                            EndsTextSegment: executionPrompt is not null);

                        foreach (ToolCall call in resolved)
                        {
                            _history.RecordToolActivity(
                                context.TurnId,
                                call,
                                new ToolResult(
                                    call.CallId,
                                    ConversationToolResultKind.Accepted,
                                    value: null));
                        }
                        yield return new TurnFinished(commitGate);
                    }
                    finally
                    {
                        // If downstream abandons the stream before the session commits, the
                        // scheduled task exits without running user code or publishing an
                        // orphaned result. After a commit this is a no-op.
                        commitGate.Abort();
                    }
                    yield break;
                }

                // The provider batch remains foreground-atomic when preferences are mixed, but a
                // locally configured background preference still admits one trusted prompt before
                // any user tool code starts. The forced boundary submits even short/unpunctuated
                // prompts to the TTS pipeline before this iterator is resumed.
                if (executionPrompt is not null)
                    yield return new TurnTextDelta(executionPrompt, EndsTextSegment: true);

                ToolResult[] results = await ExecuteBatchAsync(
                    context.TurnId,
                    resolved,
                    catalog,
                    retryBlockedAfterUnknownOutcome,
                    cancellationToken).ConfigureAwait(false);
                toolTranscript.Add(callMessage);
                for (int index = 0; index < resolved.Length; index++)
                {
                    ChatMessage resultMessage = ToToolResultMessage(results[index]);
                    transcript.Add(resultMessage);
                    toolTranscript.Add(resultMessage);
                }
            }

            throw new InvalidOperationException("The model exceeded the structured tool round limit.");
        }
        finally
        {
            pendingGate?.Abort();
        }
    }

    /// <summary>
    /// Runs one provider batch. Authorization stays strictly sequential in provider call order,
    /// because a spoken approval cannot ask two questions at once; execution may overlap for calls
    /// whose risk class carries no external side effect. Results are returned positionally, so the
    /// provider transcript order never depends on completion order.
    /// </summary>
    private async ValueTask<ToolResult[]> ExecuteBatchAsync(
        long turnId,
        ToolCall[] calls,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        CancellationToken cancellationToken)
    {
        var results = new ToolResult[calls.Length];
        var pending = new List<(int Index, ToolCall Call)>(calls.Length);
        for (int index = 0; index < calls.Length; index++)
        {
            ToolCall call = calls[index];
            ToolDefinition definition = FindDefinition(call.ToolId, catalog);
            if (ResolveBlockedReplay(turnId, call, retryBlockedAfterUnknownOutcome) is
                { } blockedReplay)
            {
                results[index] = blockedReplay;
                continue;
            }

            if (_runtime.ValidateArguments(call) is { } invalidArguments)
            {
                results[index] = RecordToolResult(turnId, call, invalidArguments);
                continue;
            }

            ToolApprovalDecision approval = await _runtime.AuthorizeAsync(call, cancellationToken)
                .ConfigureAwait(false);
            if (approval != ToolApprovalDecision.Allowed)
            {
                results[index] = RecordToolResult(
                    turnId,
                    call,
                    new ToolResult(
                        call.CallId,
                        ConversationToolResultKind.Denied,
                        null,
                        "approval_denied"));
                continue;
            }

            if (CanRunConcurrently(definition.Risk))
                pending.Add((index, call));
            else
            {
                results[index] = await ExecuteAuthorizedAsync(
                    turnId,
                    call,
                    definition,
                    retryBlockedAfterUnknownOutcome,
                    cancellationToken).ConfigureAwait(false);
            }
        }

        if (pending.Count == 0)
            return results;
        if (pending.Count == 1 || _options.MaxParallelToolCalls <= 1)
        {
            foreach ((int index, ToolCall call) in pending)
            {
                results[index] = await ExecuteAuthorizedAsync(
                    turnId,
                    call,
                    FindDefinition(call.ToolId, catalog),
                    retryBlockedAfterUnknownOutcome,
                    cancellationToken).ConfigureAwait(false);
            }
            return results;
        }

        using var concurrency = new SemaphoreSlim(
            _options.MaxParallelToolCalls,
            _options.MaxParallelToolCalls);
        Task[] executions = pending.Select(async item =>
        {
            await concurrency.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                results[item.Index] = await ExecuteAuthorizedAsync(
                    turnId,
                    item.Call,
                    FindDefinition(item.Call.ToolId, catalog),
                    retryBlockedAfterUnknownOutcome,
                    cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                concurrency.Release();
            }
        }).ToArray();
        await Task.WhenAll(executions).ConfigureAwait(false);
        return results;
    }

    /// <summary>
    /// Only risk classes without external side effects may overlap. Anything that mutates state
    /// keeps provider call order so its effects land in a deterministic sequence.
    /// </summary>
    private static bool CanRunConcurrently(ConversationToolRisk risk)
        => risk is ConversationToolRisk.ReadOnly or ConversationToolRisk.Sensitive;

    private void ScheduleBackground(
        TurnExecutionContext context,
        ToolCall[] calls,
        ChatMessage[] transcript,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        TurnCommitGate commitGate,
        List<ChatMessage> toolTranscript)
    {
        CancellationToken token = _runCancellation?.Token
            ?? throw new InvalidOperationException("The tool conversation session is not started.");
        var registered = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        Task task;
        lock (_taskGate)
        {
            Task precedingPublication = _backgroundPublicationTail;
            task = Task.Run(
                async () =>
                {
                    await registered.Task.ConfigureAwait(false);
                    await RunBackgroundAsync(
                        context,
                        calls,
                        transcript,
                        catalog,
                        retryBlockedAfterUnknownOutcome,
                        commitGate,
                        precedingPublication,
                        toolTranscript,
                        token).ConfigureAwait(false);
                },
                CancellationToken.None);
            _backgroundTasks.Add(task);
            _backgroundPublicationTail = task;
            _ = task.ContinueWith(
                completed =>
                {
                    lock (_taskGate)
                        _backgroundTasks.Remove(completed);
                },
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
        registered.TrySetResult();
    }

    private async Task RunBackgroundAsync(
        TurnExecutionContext context,
        ToolCall[] calls,
        ChatMessage[] initialTranscript,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        TurnCommitGate commitGate,
        Task precedingPublication,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        bool committed;
        try
        {
            committed = await commitGate.WaitAsync().WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return;
        }
        if (!committed)
            return;

        var transcript = initialTranscript.ToList();
        var allCalls = calls.ToList();
        var results = new List<ToolResult>();
        bool executionFailed = false;
        try
        {
            ToolResult[] batch = await ExecuteBatchAsync(
                context.TurnId,
                calls,
                catalog,
                retryBlockedAfterUnknownOutcome,
                cancellationToken).ConfigureAwait(false);
            // Persist a batch only once every call has a result. An approval failure must not
            // leave an unmatched call in the history used by later model requests.
            toolTranscript.Add(CreateAssistantToolCallMessage(calls, catalog));
            foreach (ToolResult result in batch)
            {
                results.Add(result);
                ChatMessage resultMessage = ToToolResultMessage(result);
                transcript.Add(resultMessage);
                toolTranscript.Add(resultMessage);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return;
        }
        catch
        {
            executionFailed = true;
        }

        string text = string.Empty;
        if (!executionFailed)
        {
            try
            {
                text = await CompleteBackgroundModelLoopAsync(
                    context,
                    transcript,
                    catalog,
                    allCalls,
                    results,
                    retryBlockedAfterUnknownOutcome,
                    toolTranscript,
                    cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return;
            }
            catch
            {
                // Tool execution has already resolved and may have produced side effects. A
                // summarization failure must not rewrite that outcome as an execution failure.
            }
        }

        if (string.IsNullOrWhiteSpace(text))
            text = CreateDeterministicBackgroundText(allCalls, results, catalog);

        // Commit the structured exchange and summary together before publishing the result.
        _history.RecordBackgroundResult(context.TurnId, text, toolTranscript);
        if (cancellationToken.IsCancellationRequested)
            return;

        // Submission-order FIFO: execution may overlap, but a later result never jumps ahead of
        // an earlier background acknowledgement in the speech queue.
        try { await precedingPublication.ConfigureAwait(false); } catch { }
        cancellationToken.ThrowIfCancellationRequested();
        await _backgroundOutput.PublishAsync(
            new ConversationBackgroundMessage(
                Guid.NewGuid(),
                context.ConversationId,
                context.RequestId,
                text),
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<string> CompleteBackgroundModelLoopAsync(
        TurnExecutionContext context,
        List<ChatMessage> transcript,
        ToolCatalogSnapshot catalog,
        List<ToolCall> allCalls,
        List<ToolResult> allResults,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        int totalCalls = 0;
        for (int round = 0; round < MaxModelRounds; round++)
        {
            var requested = new List<ModelTurnToolCall>();
            var visible = new List<VisibleTextDelta>();
            await foreach (ModelTurnEvent modelEvent in _client.RunAsync(
                new ModelTurnContext(
                    transcript,
                    context.Options,
                    catalog,
                    context.TurnId,
                    context.RequestId),
                cancellationToken).ConfigureAwait(false))
            {
                if (modelEvent is VisibleTextDelta { Text.Length: > 0 } delta)
                    visible.Add(delta);
                else if (modelEvent is ModelTurnToolCall call)
                    requested.Add(call);
            }

            // Apply the same final-answer rule as the foreground loop: discard text
            // accompanying tool calls and continue with their structured results.
            if (requested.Count == 0)
                return string.Concat(visible.Select(static delta => delta.Text));

            totalCalls += requested.Count;
            if (totalCalls > MaxToolCallsPerTurn)
                throw new InvalidOperationException("The background tool loop exceeded its call limit.");
            ToolCall[] calls = requested.Select(call => ResolveCall(call, catalog)).ToArray();
            allCalls.AddRange(calls);
            ChatMessage callMessage = CreateAssistantToolCallMessage(calls, catalog);
            transcript.Add(callMessage);
            ToolResult[] batch = await ExecuteBatchAsync(
                context.TurnId,
                calls,
                catalog,
                retryBlockedAfterUnknownOutcome,
                cancellationToken).ConfigureAwait(false);
            toolTranscript.Add(callMessage);
            foreach (ToolResult result in batch)
            {
                allResults.Add(result);
                ChatMessage resultMessage = ToToolResultMessage(result);
                transcript.Add(resultMessage);
                toolTranscript.Add(resultMessage);
            }
        }

        throw new InvalidOperationException("The background model loop exceeded its round limit.");
    }

    private ToolResult? ResolveBlockedReplay(long turnId, ToolCall call, UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome)
    {
        if (!retryBlockedAfterUnknownOutcome.Contains(
            new ToolInvocationFingerprint(call.ToolId, call.ArgumentsHash)))
        {
            return null;
        }

        return RecordToolResult(
            turnId,
            call,
            new ToolResult(
                call.CallId,
                ConversationToolResultKind.UnknownOutcome,
                null,
                "retry_blocked_after_unknown_outcome"));
    }

    /// <summary>
    /// Executes an already authorized call, retrying only transient failures of read-only tools.
    /// Retrying anything with a side effect is unsafe, and a tool whose outcome is unknown is
    /// blocked from replay for the rest of the turn.
    /// </summary>
    private async ValueTask<ToolResult> ExecuteAuthorizedAsync(
        long turnId,
        ToolCall call,
        ToolDefinition definition,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        CancellationToken cancellationToken)
    {
        int maxAttempts = definition.Risk == ConversationToolRisk.ReadOnly
            ? 1 + Math.Max(0, _options.MaxToolRetryAttempts)
            : 1;
        long configuredBudgetTicks = TimeSpan.FromMilliseconds(_options.MaxToolRetryBudgetMs).Ticks;
        // Compare before multiplication so a large, valid tool timeout cannot overflow.
        long budgetTicks = definition.Timeout.Ticks > configuredBudgetTicks / maxAttempts
            ? configuredBudgetTicks
            : definition.Timeout.Ticks * maxAttempts;
        TimeSpan budget = TimeSpan.FromTicks(budgetTicks);
        long startTimestamp = Stopwatch.GetTimestamp();
        ToolResult result = await ExecuteOnceAsync(
            call,
            definition,
            budget < definition.Timeout ? budget : definition.Timeout,
            cancellationToken).ConfigureAwait(false);

        for (int attempt = 1; attempt < maxAttempts && IsTransient(result); attempt++)
        {
            TimeSpan backoff = TimeSpan.FromMilliseconds(
                _options.ToolRetryBaseDelayMs * Math.Pow(3, attempt - 1));
            TimeSpan remaining = budget - Stopwatch.GetElapsedTime(startTimestamp);
            if (remaining <= backoff)
                break;

            try
            {
                await Task.Delay(backoff, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return RecordToolResult(
                    turnId,
                    call,
                    new ToolResult(
                        call.CallId,
                        ConversationToolResultKind.Cancelled,
                        null,
                        "cancelled"));
            }

            remaining = budget - Stopwatch.GetElapsedTime(startTimestamp);
            if (remaining <= TimeSpan.Zero)
                break;
            result = await ExecuteOnceAsync(
                call, definition,
                remaining < definition.Timeout ? remaining : definition.Timeout,
                cancellationToken).ConfigureAwait(false);
        }

        if (result.Kind == ConversationToolResultKind.UnknownOutcome &&
            IsSideEffecting(definition.Risk))
        {
            retryBlockedAfterUnknownOutcome.Add(
                new ToolInvocationFingerprint(call.ToolId, call.ArgumentsHash));
        }
        return RecordToolResult(turnId, call, result);
    }

    private async ValueTask<ToolResult> ExecuteOnceAsync(
        ToolCall call,
        ToolDefinition definition,
        TimeSpan executionTimeout,
        CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(executionTimeout);
        try
        {
            return await _runtime.ExecuteAsync(call, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            ConversationToolResultKind kind = IsSideEffecting(definition.Risk)
                ? ConversationToolResultKind.UnknownOutcome
                : ConversationToolResultKind.TimedOut;
            return new ToolResult(call.CallId, kind, null, "timeout");
        }
        catch (OperationCanceledException)
        {
            return new ToolResult(
                call.CallId,
                ConversationToolResultKind.Cancelled,
                null,
                "cancelled");
        }
        catch (Exception exception)
        {
            return new ToolResult(
                call.CallId,
                ConversationToolResultKind.ToolError,
                null,
                exception.GetType().Name);
        }
    }

    /// <summary>
    /// A failure worth one more attempt: the call may simply not have reached a healthy server. A
    /// timeout is deliberately excluded for anything but a read-only tool, and
    /// <see cref="ConversationToolResultKind.UnknownOutcome"/> is never transient because the work
    /// may already have happened.
    /// </summary>
    private static bool IsTransient(ToolResult result) => result.Kind switch
    {
        ConversationToolResultKind.TimedOut => true,
        ConversationToolResultKind.ToolError => IsTransientErrorCode(result.ErrorCode),
        ConversationToolResultKind.ProtocolError => IsTransientErrorCode(result.ErrorCode),
        _ => false,
    };

    private static bool IsTransientErrorCode(string? errorCode) => errorCode switch
    {
        null => false,
        nameof(HttpRequestException) => true,
        nameof(TimeoutException) => true,
        nameof(IOException) => true,
        nameof(SocketException) => true,
        "transport_error" => true,
        "server_unavailable" => true,
        "rate_limited" => true,
        _ => false,
    };

    private ToolResult RecordToolResult(long turnId, ToolCall call, ToolResult result)
    {
        _history.RecordToolActivity(turnId, call, result);
        return result;
    }

    private static bool IsSideEffecting(ConversationToolRisk risk) => risk is
        ConversationToolRisk.SideEffecting or
        ConversationToolRisk.Destructive or
        ConversationToolRisk.Privileged;

    private static string CreateDeterministicBackgroundText(
        IReadOnlyList<ToolCall> calls,
        IReadOnlyList<ToolResult> results,
        ToolCatalogSnapshot catalog)
    {
        var resultByCallId = new Dictionary<string, ToolResult>(StringComparer.Ordinal);
        foreach (ToolResult result in results)
            resultByCallId[result.CallId] = result;

        var fragments = new List<string>(calls.Count);
        foreach (ToolCall call in calls)
        {
            string title = FindDefinition(call.ToolId, catalog).Title;
            if (!resultByCallId.TryGetValue(call.CallId, out ToolResult? result))
            {
                fragments.Add($"{title}任务执行失败。");
                continue;
            }

            fragments.Add(result.Kind switch
            {
                ConversationToolResultKind.Success => $"{title}任务已完成。",
                ConversationToolResultKind.Denied => $"{title}任务未获授权，未执行。",
                ConversationToolResultKind.Cancelled => $"{title}任务已取消。",
                ConversationToolResultKind.UnknownOutcome =>
                    $"{title}任务执行状态暂时无法确认，请勿重复操作。",
                _ => $"{title}任务执行失败。",
            });
        }
        return string.Concat(fragments);
    }

    private static ToolCall ResolveCall(ModelTurnToolCall requested, ToolCatalogSnapshot catalog)
    {
        ToolDefinition definition = FindDefinition(requested.ToolId, catalog);
        (OwnedJsonValue arguments, ConversationToolExecutionPreference preference) =
            ToolExecutionPreferenceSchema.Extract(requested.Arguments, definition.ExecutionPolicy);
        return new ToolCall(
            requested.CallId,
            requested.ToolId,
            arguments,
            preference);
    }

    private static ToolDefinition FindDefinition(ConversationToolId id, ToolCatalogSnapshot catalog)
        => catalog.Definitions.FirstOrDefault(definition => definition.Id == id)
            ?? throw new InvalidOperationException($"Unknown tool '{id}'.");

    private static string? FindCustomBackgroundExecutionPrompt(IEnumerable<ToolCall> calls, ToolCatalogSnapshot catalog)
        => calls
            .Where(static call =>
                call.ExecutionPreference == ConversationToolExecutionPreference.Background)
            .Select(call => FindDefinition(call.ToolId, catalog).ExecutionPrompt)
            .FirstOrDefault(static prompt => !string.IsNullOrWhiteSpace(prompt));

    private static string CreateStandardBackgroundExecutionPrompt(IEnumerable<ToolCall> calls, ToolCatalogSnapshot catalog)
    {
        string names = string.Join("、", calls
            .Select(call => FindDefinition(call.ToolId, catalog).Title)
            .Distinct(StringComparer.Ordinal));
        return $"我正在后台执行{names}任务，等会给你结果。";
    }

    private static ChatMessage CreateAssistantToolCallMessage(IEnumerable<ToolCall> calls, ToolCatalogSnapshot catalog)
        => ChatMessage.AssistantToolCalls(calls.Select(call => new ToolCallContent(
            call.CallId,
            catalog.ExposedNames[call.ToolId],
            call.Arguments.CanonicalJson)).ToArray());

    private static ChatMessage ToToolResultMessage(ToolResult result)
    {
        return ChatMessage.ToolResult(
            result.CallId,
            result.ToModelResultJson(),
            result.Kind != ConversationToolResultKind.Success);
    }

    private void PublishToolTranscript(long turnId, List<ChatMessage> toolTranscript)
    {
        if (toolTranscript.Count == 0)
            return;

        ChatMessage[] messages = toolTranscript.ToArray();
        toolTranscript.Clear();
        _history.RecordToolExchange(turnId, messages);
    }

    private readonly record struct ToolInvocationFingerprint(ConversationToolId ToolId, string ArgumentsHash);

    /// <summary>
    /// Invocations whose outcome could not be determined and must not be replayed for the rest of
    /// the turn. Concurrent batch execution reads and writes it from several tasks, so every access
    /// is guarded.
    /// </summary>
    private sealed class UnknownOutcomeBlocklist
    {
        private readonly object _gate = new();
        private readonly HashSet<ToolInvocationFingerprint> _fingerprints = [];

        public bool Contains(ToolInvocationFingerprint fingerprint)
        {
            lock (_gate)
                return _fingerprints.Contains(fingerprint);
        }

        public void Add(ToolInvocationFingerprint fingerprint)
        {
            lock (_gate)
                _fingerprints.Add(fingerprint);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try { await StopAsync().ConfigureAwait(false); } catch { }
        try { await _runtime.DisposeAsync().ConfigureAwait(false); } catch { }
    }
}
