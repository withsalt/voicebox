using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Sockets;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Assistant.Conversation.Internal.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>
/// Runs one provider batch of tool calls: argument validation, approval, ordering, timeout, retry and audit.
/// The batch keeps control of every call. A tool that ignores its timeout or cancellation is left behind and
/// its eventual result is discarded; it cannot hold the turn.
/// </summary>
internal sealed class ToolBatchExecutor
{
    // CancellationTokenSource.CancelAfter rejects longer delays; such a timeout is treated as "no timeout".
    private static readonly TimeSpan MaxTimerDelay = TimeSpan.FromMilliseconds(uint.MaxValue - 1);
    private static readonly TimeSpan LateToolWarningThreshold = TimeSpan.FromSeconds(1);

    private readonly ConversationToolRuntime _runtime;
    private readonly ConversationHistoryStore _history;
    private readonly ConversationToolOptions _options;
    private readonly ILogger _logger;
    private readonly ConcurrentDictionary<Task, byte> _leftBehind = new();

    public ToolBatchExecutor(
        ConversationToolRuntime runtime,
        ConversationHistoryStore history,
        ConversationToolOptions options,
        ILogger logger)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Returns one outcome per call, in call order. Approval is asked strictly one call at a time in call order,
    /// because a spoken approval cannot ask two questions at once. Consecutive calls without external side effects
    /// run together; a call with side effects waits for the calls before it and finishes before the calls after
    /// it start, so the batch's effects land exactly in the order the model wrote them.
    /// </summary>
    public async Task<ToolOutcome[]> ExecuteAsync(
        long turnId,
        IReadOnlyList<ToolCall> calls,
        UnknownOutcomeBlocklist blocklist,
        CancellationToken cancellationToken)
    {
        var outcomes = new ToolOutcome[calls.Count];
        // Approved calls without side effects, held until the next side-effecting call or the end of the batch.
        var overlapping = new List<(int Index, ToolCall Call, ConversationToolCall Request)>();
        for (int index = 0; index < calls.Count; index++)
        {
            // A canceled turn starts no further call, including one that needs no approval.
            cancellationToken.ThrowIfCancellationRequested();
            ToolCall call = calls[index];
            if (Refuse(call, blocklist) is { } refused)
            {
                outcomes[index] = Record(turnId, call, refused);
                continue;
            }

            ConversationToolCall request = call.ToCall();
            if (!await _runtime.AuthorizeAsync(call, request, cancellationToken).ConfigureAwait(false))
            {
                _logger.LogInformation("Tool {Tool} call {CallId} was not approved.", call.Tool.Id, call.CallId);
                outcomes[index] = Record(turnId, call, ToolOutcome.Failure(ConversationToolResultKind.Denied, "approval_denied"));
                continue;
            }

            if (CanRunConcurrently(call.Tool.Descriptor.Risk))
            {
                overlapping.Add((index, call, request));
                continue;
            }

            await RunOverlappingAsync().ConfigureAwait(false);
            outcomes[index] = await RunAsync(turnId, call, request, blocklist, cancellationToken).ConfigureAwait(false);
        }

        await RunOverlappingAsync().ConfigureAwait(false);
        // Calls that were cut short report Cancelled instead of throwing; the turn itself ends here.
        cancellationToken.ThrowIfCancellationRequested();
        return outcomes;

        async Task RunOverlappingAsync()
        {
            if (overlapping.Count == 0)
                return;

            var parallel = new ParallelOptions
            {
                MaxDegreeOfParallelism = _options.MaxParallelToolCalls,
                CancellationToken = cancellationToken,
            };
            await Parallel.ForEachAsync(
                overlapping,
                parallel,
                async (item, _) => outcomes[item.Index] = await RunAsync(turnId, item.Call, item.Request, blocklist, cancellationToken)
                    .ConfigureAwait(false)).ConfigureAwait(false);
            overlapping.Clear();
        }
    }

    /// <summary>Outcomes decided without running the tool: a blocked replay or arguments the model must correct.</summary>
    private ToolOutcome? Refuse(ToolCall call, UnknownOutcomeBlocklist blocklist)
    {
        if (blocklist.Contains(call))
            return ToolOutcome.Failure(ConversationToolResultKind.UnknownOutcome, "retry_blocked_after_unknown_outcome");

        if (_runtime.ValidateArguments(call) is not { } violations)
            return null;

        _logger.LogDebug("Tool {Tool} call {CallId} was rejected: its arguments do not match the input schema.", call.Tool.Id, call.CallId);
        return ToolOutcome.From(ConversationToolResult.Failure(
            ConversationToolResultKind.InputRequired,
            "schema_validation_failed",
            structuredContentJson: violations));
    }

    /// <summary>
    /// Executes an approved call. The first attempt is bounded by the tool's own timeout alone. Only a read-only
    /// tool is retried, only after a timeout or a failure marked transient, and only while the retry budget lasts.
    /// A side-effecting call whose outcome is unknown is blocked from replay for the rest of the turn.
    /// </summary>
    private async Task<ToolOutcome> RunAsync(
        long turnId,
        ToolCall call,
        ConversationToolCall request,
        UnknownOutcomeBlocklist blocklist,
        CancellationToken cancellationToken)
    {
        ConversationToolDescriptor descriptor = call.Tool.Descriptor;
        int maxAttempts = descriptor.Risk == ConversationToolRisk.ReadOnly ? 1 + _options.MaxToolRetryAttempts : 1;
        long started = Stopwatch.GetTimestamp();

        int attempts = 1;
        ToolOutcome outcome = await RunOnceAsync(call, request, descriptor.Timeout, cancellationToken).ConfigureAwait(false);
        while (attempts < maxAttempts && IsRetryable(outcome.Result))
        {
            // A retry starts only if its backoff still fits the budget, and may only use what is left of it.
            double backoffMs = _options.ToolRetryBaseDelayMs * Math.Pow(3, attempts - 1);
            double remainingMs = _options.MaxToolRetryBudgetMs - Stopwatch.GetElapsedTime(started).TotalMilliseconds - backoffMs;
            // Written as a negation so a backoff that overflowed to infinity or NaN also ends the retries.
            if (!(remainingMs > 0))
                break;

            _logger.LogDebug(
                "Retrying tool {Tool} call {CallId} after {Kind} ({ErrorCode}); attempt {Attempt} of {MaxAttempts}.",
                call.Tool.Id, call.CallId, outcome.Result.Kind, outcome.Result.ErrorCode, attempts + 1, maxAttempts);
            try
            {
                await Task.Delay(TimeSpan.FromMilliseconds(backoffMs), cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                outcome = ToolOutcome.Failure(ConversationToolResultKind.Cancelled, "cancelled");
                break;
            }

            attempts++;
            TimeSpan remaining = TimeSpan.FromMilliseconds(remainingMs);
            TimeSpan timeout = remaining < descriptor.Timeout ? remaining : descriptor.Timeout;
            outcome = await RunOnceAsync(call, request, timeout, cancellationToken).ConfigureAwait(false);
        }

        ConversationToolResult result = outcome.Result;
        if (result.Kind == ConversationToolResultKind.UnknownOutcome && IsSideEffecting(descriptor.Risk))
            blocklist.Add(call);

        _logger.Log(
            result.IsSuccess || result.Kind == ConversationToolResultKind.Cancelled ? LogLevel.Debug : LogLevel.Warning,
            "Tool {Tool} call {CallId} ended as {Kind} ({ErrorCode}) after {ElapsedMs} ms and {Attempts} attempt(s).",
            call.Tool.Id, call.CallId, result.Kind, result.ErrorCode, (long)Stopwatch.GetElapsedTime(started).TotalMilliseconds, attempts);
        return Record(turnId, call, outcome);
    }

    private async Task<ToolOutcome> RunOnceAsync(
        ToolCall call,
        ConversationToolCall request,
        TimeSpan timeout,
        CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
            return ToolOutcome.Failure(ConversationToolResultKind.Cancelled, "cancelled");

        using var attempt = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (timeout < MaxTimerDelay)
            attempt.CancelAfter(timeout);

        // The tool runs on the thread pool, so even one that blocks before its first await cannot hold this turn.
        CancellationToken attemptToken = attempt.Token;
        Task<ConversationToolResult> invocation = Task.Run(() => InvokeAsync(request, attemptToken), CancellationToken.None);
        // Stop waiting as soon as the attempt is canceled, whether or not the tool has returned by then.
        await ((Task)invocation.WaitAsync(attemptToken)).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);

        if (invocation.IsCompletedSuccessfully)
        {
            try
            {
                return ToolOutcome.From(invocation.Result);
            }
            catch (JsonException exception)
            {
                _logger.LogWarning(
                    exception,
                    "Tool {Tool} call {CallId} returned structured content that is not valid JSON.",
                    call.Tool.Id, call.CallId);
                return ToolOutcome.Failure(ConversationToolResultKind.ProtocolError, "invalid_structured_content");
            }
        }

        if (!invocation.IsCompleted)
        {
            ObserveLateCompletion(invocation, call);
        }
        else if (invocation.Exception?.InnerException is { } failure and not OperationCanceledException)
        {
            _logger.LogWarning(failure, "Tool {Tool} call {CallId} threw.", call.Tool.Id, call.CallId);
            return ToolOutcome.Failure(ConversationToolResultKind.ToolError, failure.GetType().Name, IsTransientException(failure));
        }

        // The tool is still running, or it stopped because it was canceled.
        if (cancellationToken.IsCancellationRequested)
            return ToolOutcome.Failure(ConversationToolResultKind.Cancelled, "cancelled");

        // The timeout elapsed, or the tool gave up on its own. After dispatch, a timeout cannot prove that a
        // mutation did not happen.
        return IsSideEffecting(call.Tool.Descriptor.Risk)
            ? ToolOutcome.Failure(ConversationToolResultKind.UnknownOutcome, "timeout")
            : ToolOutcome.Failure(ConversationToolResultKind.TimedOut, "timeout", isTransient: true);
    }

    private async Task<ConversationToolResult> InvokeAsync(ConversationToolCall request, CancellationToken cancellationToken)
        => await _runtime.InvokeAsync(request, cancellationToken).ConfigureAwait(false)
            ?? throw new InvalidOperationException("The tool returned a null result.");

    /// <summary>
    /// Waits a bounded time for calls that were left behind after a timeout or cancellation, so a tool that is
    /// still unwinding is not disposed underneath itself. A tool that never returns does not block shutdown.
    /// </summary>
    public async Task WaitForLeftBehindCallsAsync(TimeSpan gracePeriod)
    {
        Task[] pending = _leftBehind.Keys.ToArray();
        if (pending.Length == 0)
            return;

        // The tracked tasks only report; they never fail, so the wait ends by completion or by the grace period.
        await ((Task)Task.WhenAll(pending).WaitAsync(gracePeriod)).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
        int stillRunning = pending.Count(static task => !task.IsCompleted);
        if (stillRunning > 0)
            _logger.LogWarning("{Count} tool call(s) ignored cancellation and are still running; shutdown continues.", stillRunning);

        // Each call gets one grace period: stop followed by dispose must not wait for the same call twice.
        foreach (Task task in pending)
            _leftBehind.TryRemove(task, out _);
    }

    /// <summary>Tracks a call that was left behind until it returns, so shutdown can give it a moment to unwind.</summary>
    private void ObserveLateCompletion(Task invocation, ToolCall call)
    {
        Task report = ReportLateCompletionAsync(invocation, call, Stopwatch.GetTimestamp());
        _leftBehind.TryAdd(report, 0);
        _ = report.ContinueWith(
            static (task, state) => ((ConcurrentDictionary<Task, byte>)state!).TryRemove(task, out _),
            _leftBehind,
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }

    /// <summary>Keeps a late failure from going unobserved, and reports how late the call finished.</summary>
    private async Task ReportLateCompletionAsync(Task invocation, ToolCall call, long leftBehind)
    {
        await invocation.ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
        TimeSpan late = Stopwatch.GetElapsedTime(leftBehind);
        _logger.Log(
            late > LateToolWarningThreshold ? LogLevel.Warning : LogLevel.Debug,
            invocation.Exception?.InnerException,
            "Tool {Tool} call {CallId} finished {LateMs} ms after it was asked to stop; its result was discarded.",
            call.Tool.Id, call.CallId, (long)late.TotalMilliseconds);
    }

    private ToolOutcome Record(long turnId, ToolCall call, ToolOutcome outcome)
    {
        _history.RecordToolActivity(turnId, call.CallId, call.Tool.Id, outcome.Result.Kind);
        return outcome;
    }

    /// <summary>A failure worth one more attempt. An unknown outcome is never retried: the work may already have happened.</summary>
    private static bool IsRetryable(ConversationToolResult result)
        => result.Kind != ConversationToolResultKind.UnknownOutcome &&
            (result.IsTransient || result.Kind == ConversationToolResultKind.TimedOut);

    /// <summary>
    /// A thrown exception carries no retry flag, so only connection-level failures count as transient. A broken
    /// network stream throws <see cref="IOException"/> itself; its file-system subclasses such as
    /// <see cref="FileNotFoundException"/> describe a permanent condition and are not retried.
    /// </summary>
    private static bool IsTransientException(Exception exception)
        => exception is HttpRequestException or TimeoutException or SocketException || exception.GetType() == typeof(IOException);

    /// <summary>
    /// Only risk classes without external side effects may overlap. Anything that mutates state keeps provider
    /// call order so its effects land in a deterministic sequence.
    /// </summary>
    private static bool CanRunConcurrently(ConversationToolRisk risk)
        => risk is ConversationToolRisk.ReadOnly or ConversationToolRisk.Sensitive;

    private static bool IsSideEffecting(ConversationToolRisk risk)
        => risk is ConversationToolRisk.SideEffecting or ConversationToolRisk.Destructive or ConversationToolRisk.Privileged;
}

/// <summary>
/// Invocations whose outcome could not be determined and must not be replayed for the rest of the turn.
/// Overlapping calls of one batch read and write it from several tasks, so every access is guarded.
/// </summary>
internal sealed class UnknownOutcomeBlocklist
{
    private readonly object _gate = new();
    private readonly HashSet<(ConversationToolId ToolId, string ArgumentsJson)> _invocations = [];

    public bool Contains(ToolCall call)
    {
        lock (_gate)
            return _invocations.Contains((call.Tool.Id, call.ArgumentsJson));
    }

    public void Add(ToolCall call)
    {
        lock (_gate)
            _invocations.Add((call.Tool.Id, call.ArgumentsJson));
    }
}
