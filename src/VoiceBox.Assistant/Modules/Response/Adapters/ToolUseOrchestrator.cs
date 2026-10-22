using System.Diagnostics;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Core.Services.LLMService.Models;
using InternalToolRisk = VoiceBox.Assistant.Modules.ModelInteraction.Models.AssistantToolRisk;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>
/// Executes a bounded native tool loop and keeps provider transcript, user-visible text and tool
/// execution separate. Background calls return a fixed trusted acknowledgement immediately.
/// </summary>
internal sealed class ToolUseOrchestrator :
    IResponseGenerator,
    IResponseGeneratorLifecycle,
    IBackgroundToolResultSource,
    IToolHistorySource,
    IDisposable
{
    private const int MaxModelRounds = 8;
    private const int MaxToolCallsPerTurn = 16;

    private readonly ILlmTurnClient _client;
    private readonly IAssistantToolRuntime _runtime;
    private readonly VoiceAssistantToolOptions _options;
    private readonly object _taskGate = new();
    private readonly HashSet<Task> _backgroundTasks = [];
    private Task _backgroundPublicationTail = Task.CompletedTask;
    private CancellationTokenSource? _runCancellation;
    private ToolCatalogSnapshot? _catalog;
    private IBackgroundToolResultSink? _backgroundResultSink;
    private IToolHistorySink? _toolHistorySink;
    private int _disposed;

    public ToolUseOrchestrator(
        ILlmTurnClient client,
        IAssistantToolRuntime runtime,
        VoiceAssistantToolOptions options)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_runCancellation is not null)
            throw new InvalidOperationException("Tool response generator is already started.");

        await _runtime.StartAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ToolCatalogSnapshot catalog = await _runtime.GetSnapshotAsync(
                new ToolRequestContext(0),
                cancellationToken).ConfigureAwait(false);
            if (catalog.IsEmpty)
                throw new InvalidOperationException("UseMcpTools did not expose any tools.");
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

    public void CancelBackgroundOperations()
    {
        CancellationTokenSource? run = _runCancellation;
        if (run is null)
            return;
        try { run.Cancel(); } catch (ObjectDisposedException) { }
    }

    public void BindBackgroundResultSink(IBackgroundToolResultSink sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (Interlocked.CompareExchange(ref _backgroundResultSink, sink, null) is not null)
            throw new InvalidOperationException("Background tool result sink is already bound.");
    }

    public void BindToolHistorySink(IToolHistorySink sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (Interlocked.CompareExchange(ref _toolHistorySink, sink, null) is not null)
            throw new InvalidOperationException("Tool history sink is already bound.");
    }

    public async IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
        ModelTurnRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ToolCatalogSnapshot catalog = _catalog
            ?? throw new InvalidOperationException("Tool response generator is not started.");

        var transcript = request.Transcript.ToList();
        var executedCalls = new List<AssistantToolCall>();
        var executedResults = new List<AssistantToolResult>();
        var retryBlockedAfterUnknownOutcome = new UnknownOutcomeBlocklist();
        // Provider transcript slice this turn produced, reported so a later turn can still observe
        // structured tool results instead of only the spoken answer.
        var toolTranscript = new List<ChatMessage>();
        ModelUsage totalUsage = default;
        int totalToolCalls = 0;
        for (int round = 0; round < MaxModelRounds; round++)
        {
            var calls = new List<ToolCallRequested>();
            var visibleText = new List<VisibleTextDelta>();
            ModelTurnCompleted? completed = null;
            var roundRequest = new ModelTurnRequest(
                transcript,
                request.Options,
                catalog,
                request.TurnId);
            await foreach (ModelTurnEvent modelEvent in _client.RunAsync(
                roundRequest,
                cancellationToken).ConfigureAwait(false))
            {
                switch (modelEvent)
                {
                    case ToolCallRequested call:
                        calls.Add(call);
                        break;
                    case VisibleTextDelta { Text.Length: > 0 } delta:
                        visibleText.Add(delta);
                        break;
                    case VisibleTextDelta:
                        break;
                    case ModelTurnCompleted value:
                        completed = value;
                        break;
                    default:
                        yield return modelEvent;
                        break;
                }
            }

            if (completed is not null)
                totalUsage = AddUsage(totalUsage, completed.Usage);

            // Providers may emit assistant text alongside native tool calls. Keep that
            // intermediate narration out of speech and history; only a round without
            // tool calls supplies the final visible answer.
            if (calls.Count == 0)
            {
                PublishToolTranscript(request.TurnId, toolTranscript);
                foreach (VisibleTextDelta delta in visibleText)
                    yield return delta;
                yield return new ModelTurnCompleted(totalUsage, executedCalls, executedResults);
                yield break;
            }

            totalToolCalls += calls.Count;
            if (totalToolCalls > MaxToolCallsPerTurn)
                throw new InvalidOperationException("The model exceeded the per-turn tool call limit.");

            AssistantToolCall[] resolved = calls.Select(call => ResolveCall(call, catalog)).ToArray();
            // A provider tool-call batch must be resolved atomically. If it mixes background and
            // foreground preferences, foreground wins so every tool result can be returned in the
            // same provider transcript without leaving dangling call ids.
            bool runInBackground = resolved.All(static call =>
                call.ExecutionPreference == AssistantToolExecutionPreference.Background);
            string? executionPrompt = FindCustomBackgroundExecutionPrompt(resolved, catalog);
            ChatMessage callMessage = CreateAssistantToolCallMessage(resolved, catalog);
            transcript.Add(callMessage);
            toolTranscript.Add(callMessage);

            if (runInBackground)
            {
                var acknowledgementAccepted = new TaskCompletionSource(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                ScheduleBackground(
                    request.TurnId,
                    resolved,
                    transcript.ToArray(),
                    request.Options,
                    catalog,
                    retryBlockedAfterUnknownOutcome,
                    acknowledgementAccepted.Task,
                    toolTranscript);
                try
                {
                    TaskCompletionSource? speechBoundaryCompletion = executionPrompt is null
                        ? null
                        : new TaskCompletionSource(
                            TaskCreationOptions.RunContinuationsAsynchronously);
                    yield return new VisibleTextDelta(
                        executionPrompt ?? CreateStandardBackgroundExecutionPrompt(resolved, catalog),
                        ForceSpeechBoundary: executionPrompt is not null,
                        SpeechBoundaryCompletion: speechBoundaryCompletion);
                    yield return new ModelTurnCompleted(
                        totalUsage,
                        resolved,
                        resolved.Select(call => new AssistantToolResult(
                            call.CallId,
                            AssistantToolResultKind.Accepted,
                            value: null)));

                    foreach (AssistantToolCall call in resolved)
                    {
                        _toolHistorySink?.RecordToolActivity(
                            request.TurnId,
                            call,
                            new AssistantToolResult(
                                call.CallId,
                                AssistantToolResultKind.Accepted,
                                value: null));
                    }
                    acknowledgementAccepted.TrySetResult();
                }
                finally
                {
                    // If downstream abandons the stream before accepting ModelTurnCompleted, the
                    // scheduled task exits without running user code or publishing an orphaned
                    // result. After normal resumption TrySetCanceled is a harmless no-op.
                    acknowledgementAccepted.TrySetCanceled();
                }
                yield break;
            }

            // The provider batch remains foreground-atomic when preferences are mixed, but a
            // locally configured background preference still admits one trusted prompt before
            // any user tool code starts. The forced boundary submits even short/unpunctuated
            // prompts to the TTS pipeline before this iterator is resumed.
            if (executionPrompt is not null)
            {
                var speechBoundaryCompletion = new TaskCompletionSource(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                yield return new VisibleTextDelta(
                    executionPrompt,
                    ForceSpeechBoundary: true,
                    SpeechBoundaryCompletion: speechBoundaryCompletion);
            }

            AssistantToolResult[] results = await ExecuteBatchAsync(
                request.TurnId,
                resolved,
                catalog,
                retryBlockedAfterUnknownOutcome,
                cancellationToken).ConfigureAwait(false);
            for (int index = 0; index < resolved.Length; index++)
            {
                executedCalls.Add(resolved[index]);
                executedResults.Add(results[index]);
                ChatMessage resultMessage = ToToolResultMessage(results[index]);
                transcript.Add(resultMessage);
                toolTranscript.Add(resultMessage);
            }
        }

        throw new InvalidOperationException("The model exceeded the structured tool round limit.");
    }

    /// <summary>
    /// Runs one provider batch. Authorization stays strictly sequential in provider call order,
    /// because a spoken approval cannot ask two questions at once; execution may overlap for calls
    /// whose risk class carries no external side effect. Results are returned positionally, so the
    /// provider transcript order never depends on completion order.
    /// </summary>
    private async ValueTask<AssistantToolResult[]> ExecuteBatchAsync(
        long turnId,
        AssistantToolCall[] calls,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        CancellationToken cancellationToken)
    {
        var results = new AssistantToolResult[calls.Length];
        var pending = new List<(int Index, AssistantToolCall Call)>(calls.Length);
        for (int index = 0; index < calls.Length; index++)
        {
            AssistantToolCall call = calls[index];
            AssistantToolDefinition definition = FindDefinition(call.ToolId, catalog);
            if (ResolveBlockedReplay(turnId, call, retryBlockedAfterUnknownOutcome) is
                { } blockedReplay)
            {
                results[index] = blockedReplay;
                continue;
            }

            ToolApprovalDecision approval = await _runtime.AuthorizeAsync(call, cancellationToken)
                .ConfigureAwait(false);
            if (approval != ToolApprovalDecision.Allowed)
            {
                results[index] = RecordToolResult(
                    turnId,
                    call,
                    new AssistantToolResult(
                        call.CallId,
                        AssistantToolResultKind.Denied,
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
            foreach ((int index, AssistantToolCall call) in pending)
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
    private static bool CanRunConcurrently(InternalToolRisk risk)
        => risk is InternalToolRisk.ReadOnly or InternalToolRisk.Sensitive;

    private void ScheduleBackground(
        long turnId,
        AssistantToolCall[] calls,
        ChatMessage[] transcript,
        ModelGenerationOptions options,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        Task acknowledgementAccepted,
        List<ChatMessage> toolTranscript)
    {
        CancellationToken token = _runCancellation?.Token
            ?? throw new InvalidOperationException("Tool response generator is not started.");
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
                        turnId,
                        calls,
                        transcript,
                        options,
                        catalog,
                        retryBlockedAfterUnknownOutcome,
                        acknowledgementAccepted,
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
        long turnId,
        AssistantToolCall[] calls,
        ChatMessage[] initialTranscript,
        ModelGenerationOptions options,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        Task acknowledgementAccepted,
        Task precedingPublication,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        try
        {
            await acknowledgementAccepted.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return;
        }

        var transcript = initialTranscript.ToList();
        var allCalls = calls.ToList();
        var results = new List<AssistantToolResult>();
        bool executionFailed = false;
        try
        {
            AssistantToolResult[] batch = await ExecuteBatchAsync(
                turnId,
                calls,
                catalog,
                retryBlockedAfterUnknownOutcome,
                cancellationToken).ConfigureAwait(false);
            foreach (AssistantToolResult result in batch)
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
                    turnId,
                    transcript,
                    options,
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

        // Report the exchange before the summary is published, so the turn record already carries
        // the tool messages when the sink commits its background text into history.
        PublishToolTranscript(turnId, toolTranscript);

        IBackgroundToolResultSink? sink = _backgroundResultSink;
        if (sink is null || cancellationToken.IsCancellationRequested)
            return;

        // Submission-order FIFO: execution may overlap, but a later result never jumps ahead of
        // an earlier background acknowledgement in the speech queue.
        try { await precedingPublication.ConfigureAwait(false); } catch { }
        cancellationToken.ThrowIfCancellationRequested();
        await sink.PublishAsync(turnId, text, cancellationToken).ConfigureAwait(false);
    }

    private async Task<string> CompleteBackgroundModelLoopAsync(
        long turnId,
        List<ChatMessage> transcript,
        ModelGenerationOptions options,
        ToolCatalogSnapshot catalog,
        List<AssistantToolCall> allCalls,
        List<AssistantToolResult> allResults,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        int totalCalls = 0;
        for (int round = 0; round < MaxModelRounds; round++)
        {
            var requested = new List<ToolCallRequested>();
            var visible = new List<VisibleTextDelta>();
            await foreach (ModelTurnEvent modelEvent in _client.RunAsync(
                new ModelTurnRequest(transcript, options, catalog, turnId),
                cancellationToken).ConfigureAwait(false))
            {
                if (modelEvent is VisibleTextDelta { Text.Length: > 0 } delta)
                    visible.Add(delta);
                else if (modelEvent is ToolCallRequested call)
                    requested.Add(call);
            }

            // Apply the same final-answer rule as the foreground loop: discard text
            // accompanying tool calls and continue with their structured results.
            if (requested.Count == 0)
                return string.Concat(visible.Select(static delta => delta.Text));

            totalCalls += requested.Count;
            if (totalCalls > MaxToolCallsPerTurn)
                throw new InvalidOperationException("The background tool loop exceeded its call limit.");
            AssistantToolCall[] calls = requested.Select(call => ResolveCall(call, catalog)).ToArray();
            allCalls.AddRange(calls);
            ChatMessage callMessage = CreateAssistantToolCallMessage(calls, catalog);
            transcript.Add(callMessage);
            toolTranscript.Add(callMessage);
            AssistantToolResult[] batch = await ExecuteBatchAsync(
                turnId,
                calls,
                catalog,
                retryBlockedAfterUnknownOutcome,
                cancellationToken).ConfigureAwait(false);
            foreach (AssistantToolResult result in batch)
            {
                allResults.Add(result);
                ChatMessage resultMessage = ToToolResultMessage(result);
                transcript.Add(resultMessage);
                toolTranscript.Add(resultMessage);
            }
        }

        throw new InvalidOperationException("The background model loop exceeded its round limit.");
    }

    /// <summary>
    /// Resolves a replay of an invocation whose earlier outcome was unknown; re-running it could
    /// duplicate an external side effect that may already have happened. Returns null when the call
    /// is free to proceed.
    /// </summary>
    private AssistantToolResult? ResolveBlockedReplay(
        long turnId,
        AssistantToolCall call,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome)
    {
        if (!retryBlockedAfterUnknownOutcome.Contains(
            new ToolInvocationFingerprint(call.ToolId, call.ArgumentsHash)))
        {
            return null;
        }

        return RecordToolResult(
            turnId,
            call,
            new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.UnknownOutcome,
                null,
                "retry_blocked_after_unknown_outcome"));
    }

    /// <summary>
    /// Executes an already authorized call, retrying only transient failures of read-only tools.
    /// Retrying anything with a side effect is unsafe, and a tool whose outcome is unknown is
    /// blocked from replay for the rest of the turn.
    /// </summary>
    private async ValueTask<AssistantToolResult> ExecuteAuthorizedAsync(
        long turnId,
        AssistantToolCall call,
        AssistantToolDefinition definition,
        UnknownOutcomeBlocklist retryBlockedAfterUnknownOutcome,
        CancellationToken cancellationToken)
    {
        int maxAttempts = definition.Risk == InternalToolRisk.ReadOnly
            ? 1 + Math.Max(0, _options.MaxToolRetryAttempts)
            : 1;
        long configuredBudgetTicks = TimeSpan.FromMilliseconds(_options.MaxToolRetryBudgetMs).Ticks;
        // Compare before multiplication so a large, valid tool timeout cannot overflow.
        long budgetTicks = definition.Timeout.Ticks > configuredBudgetTicks / maxAttempts
            ? configuredBudgetTicks
            : definition.Timeout.Ticks * maxAttempts;
        TimeSpan budget = TimeSpan.FromTicks(budgetTicks);
        long startTimestamp = Stopwatch.GetTimestamp();
        AssistantToolResult result = await ExecuteOnceAsync(
            call,
            definition,
            budget < definition.Timeout ? budget : definition.Timeout,
            cancellationToken).ConfigureAwait(false);

        for (int attempt = 1; attempt < maxAttempts && IsTransient(result); attempt++)
        {
            TimeSpan backoff = TimeSpan.FromMilliseconds(
                _options.ToolRetryBaseDelayMs * Math.Pow(3, attempt - 1));
            TimeSpan remaining = budget -
                Stopwatch.GetElapsedTime(startTimestamp);
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
                    new AssistantToolResult(
                        call.CallId,
                        AssistantToolResultKind.Cancelled,
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

        if (result.Kind == AssistantToolResultKind.UnknownOutcome && IsSideEffecting(definition.Risk))
        {
            retryBlockedAfterUnknownOutcome.Add(
                new ToolInvocationFingerprint(call.ToolId, call.ArgumentsHash));
        }
        return RecordToolResult(turnId, call, result);
    }

    private async ValueTask<AssistantToolResult> ExecuteOnceAsync(
        AssistantToolCall call,
        AssistantToolDefinition definition,
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
            AssistantToolResultKind kind = IsSideEffecting(definition.Risk)
                ? AssistantToolResultKind.UnknownOutcome
                : AssistantToolResultKind.TimedOut;
            return new AssistantToolResult(call.CallId, kind, null, "timeout");
        }
        catch (OperationCanceledException)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.Cancelled,
                null,
                "cancelled");
        }
        catch (Exception exception)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.ToolError,
                null,
                exception.GetType().Name);
        }
    }

    /// <summary>
    /// A failure worth one more attempt: the call may simply not have reached a healthy server. A
    /// timeout is deliberately excluded for anything but a read-only tool, and
    /// <see cref="AssistantToolResultKind.UnknownOutcome"/> is never transient because the work may
    /// already have happened.
    /// </summary>
    private static bool IsTransient(AssistantToolResult result) => result.Kind switch
    {
        AssistantToolResultKind.TimedOut => true,
        AssistantToolResultKind.ToolError => IsTransientErrorCode(result.ErrorCode),
        AssistantToolResultKind.ProtocolError => IsTransientErrorCode(result.ErrorCode),
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

    private AssistantToolResult RecordToolResult(
        long turnId,
        AssistantToolCall call,
        AssistantToolResult result)
    {
        _toolHistorySink?.RecordToolActivity(turnId, call, result);
        return result;
    }

    private static bool IsSideEffecting(InternalToolRisk risk) => risk is
        InternalToolRisk.SideEffecting or
        InternalToolRisk.Destructive or
        InternalToolRisk.Privileged;

    private static string CreateDeterministicBackgroundText(
        IReadOnlyList<AssistantToolCall> calls,
        IReadOnlyList<AssistantToolResult> results,
        ToolCatalogSnapshot catalog)
    {
        var resultByCallId = new Dictionary<string, AssistantToolResult>(StringComparer.Ordinal);
        foreach (AssistantToolResult result in results)
            resultByCallId[result.CallId] = result;

        var fragments = new List<string>(calls.Count);
        foreach (AssistantToolCall call in calls)
        {
            string title = FindDefinition(call.ToolId, catalog).Title;
            if (!resultByCallId.TryGetValue(call.CallId, out AssistantToolResult? result))
            {
                fragments.Add($"{title}任务执行失败。");
                continue;
            }

            fragments.Add(result.Kind switch
            {
                AssistantToolResultKind.Success => $"{title}任务已完成。",
                AssistantToolResultKind.Denied => $"{title}任务未获授权，未执行。",
                AssistantToolResultKind.Cancelled => $"{title}任务已取消。",
                AssistantToolResultKind.UnknownOutcome =>
                    $"{title}任务执行状态暂时无法确认，请勿重复操作。",
                _ => $"{title}任务执行失败。",
            });
        }
        return string.Concat(fragments);
    }

    private static AssistantToolCall ResolveCall(
        ToolCallRequested requested,
        ToolCatalogSnapshot catalog)
    {
        AssistantToolDefinition definition = FindDefinition(requested.ToolId, catalog);
        (OwnedJsonValue arguments, AssistantToolExecutionPreference preference) =
            ToolExecutionPreferenceSchema.Extract(requested.Arguments, definition.ExecutionPolicy);
        return new AssistantToolCall(
            requested.CallId,
            requested.ToolId,
            arguments,
            preference);
    }

    private static AssistantToolDefinition FindDefinition(
        AssistantToolId id,
        ToolCatalogSnapshot catalog)
        => catalog.Definitions.FirstOrDefault(definition => definition.Id == id)
            ?? throw new InvalidOperationException($"Unknown tool '{id.ServerAlias}/{id.ToolName}'.");

    private static string? FindCustomBackgroundExecutionPrompt(
        IEnumerable<AssistantToolCall> calls,
        ToolCatalogSnapshot catalog)
        => calls
            .Where(static call =>
                call.ExecutionPreference == AssistantToolExecutionPreference.Background)
            .Select(call => FindDefinition(call.ToolId, catalog).ExecutionPrompt)
            .FirstOrDefault(static prompt => !string.IsNullOrWhiteSpace(prompt));

    private static string CreateStandardBackgroundExecutionPrompt(
        IEnumerable<AssistantToolCall> calls,
        ToolCatalogSnapshot catalog)
    {
        string names = string.Join("、", calls
            .Select(call => FindDefinition(call.ToolId, catalog).Title)
            .Distinct(StringComparer.Ordinal));
        return $"我正在后台执行{names}任务，等会给你结果。";
    }

    private static ChatMessage CreateAssistantToolCallMessage(
        IEnumerable<AssistantToolCall> calls,
        ToolCatalogSnapshot catalog)
        => ChatMessage.AssistantToolCalls(calls.Select(call => new ToolCallContent(
            call.CallId,
            catalog.ExposedNames[call.ToolId],
            call.Arguments.CanonicalJson)).ToArray());

    private static ChatMessage ToToolResultMessage(AssistantToolResult result)
    {
        return ChatMessage.ToolResult(
            result.CallId,
            result.ToModelResultJson(),
            result.Kind != AssistantToolResultKind.Success);
    }

    private static ModelUsage AddUsage(ModelUsage left, ModelUsage right) => new(
        checked(left.PromptTokens + right.PromptTokens),
        checked(left.CompletionTokens + right.CompletionTokens),
        checked(left.TotalTokens + right.TotalTokens));

    private void PublishToolTranscript(long turnId, List<ChatMessage> toolTranscript)
    {
        if (toolTranscript.Count == 0)
            return;

        IToolHistorySink? sink = _toolHistorySink;
        if (sink is null)
            return;

        ChatMessage[] messages = toolTranscript.ToArray();
        toolTranscript.Clear();
        sink.RecordToolExchange(turnId, messages);
    }

    private readonly record struct ToolInvocationFingerprint(
        AssistantToolId ToolId,
        string ArgumentsHash);

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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try { StopAsync().GetAwaiter().GetResult(); } catch { }
        try { _runtime.DisposeAsync().AsTask().GetAwaiter().GetResult(); } catch { }
    }
}
