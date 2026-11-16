using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Internal.History;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>
/// Executes a bounded native tool loop. Visible model text is streamed as it arrives, tool batches run through
/// <see cref="ToolBatchExecutor"/>, and a background batch acknowledges with a fixed trusted text and publishes
/// its result after the foreground turn has committed.
/// </summary>
internal sealed class ToolTurnExecutor : ITurnExecutor
{
    private const int MaxModelRounds = 8;
    private const int MaxToolCallsPerTurn = 16;

    /// <summary>Keeps text spoken before a tool batch from running into whatever is said after it.</summary>
    private const string SegmentSeparator = "\n";

    private static readonly TimeSpan LeftBehindCallGracePeriod = TimeSpan.FromSeconds(2);

    private readonly StructuredLlmTurnClient _client;
    private readonly ConversationToolRuntime _runtime;
    private readonly ToolBatchExecutor _batches;
    private readonly ConversationHistoryStore _history;
    private readonly IBackgroundResultPublisher _backgroundOutput;
    private readonly ConversationToolAnnouncements _announcements;
    private readonly ILogger _logger;
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
        ConversationToolOptions options,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _history = history ?? throw new ArgumentNullException(nameof(history));
        _backgroundOutput = backgroundOutput ?? throw new ArgumentNullException(nameof(backgroundOutput));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _announcements = options.Announcements;
        _batches = new ToolBatchExecutor(runtime, history, options, logger);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_runCancellation is not null)
            throw new InvalidOperationException("The tool conversation session is already started.");

        _catalog = await _runtime.StartAsync(cancellationToken).ConfigureAwait(false);
        _runCancellation = new CancellationTokenSource();
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
        // Background tasks report their own failures.
        try { await Task.WhenAll(tasks).ConfigureAwait(false); } catch { }
        await _batches.WaitForLeftBehindCallsAsync(LeftBehindCallGracePeriod).ConfigureAwait(false);
        try { await _runtime.StopAsync().ConfigureAwait(false); } finally { run?.Dispose(); }
        _catalog = null;
    }

    public async IAsyncEnumerable<TurnEvent> ExecuteAsync(
        TurnExecutionContext context,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ToolCatalogSnapshot catalog = _catalog ?? throw new InvalidOperationException("The tool conversation session is not started.");

        TurnCommitGate? pendingGate = null;
        var transcript = context.Prompt.ToList();
        var blocklist = new UnknownOutcomeBlocklist();
        // Provider transcript slice this turn produced, reported so a later turn can still observe
        // structured tool results instead of only the spoken answer.
        var toolTranscript = new List<ChatMessage>();
        int totalToolCalls = 0;
        // Text already sent to the user stopped right before a tool batch, so whatever is said next starts on a
        // new line instead of running into it. The separator is only emitted together with that next text.
        bool separatorPending = false;
        try
        {
            for (int round = 0; round < MaxModelRounds; round++)
            {
                var calls = new List<ToolCall>();
                // Text of this round that has reached the user. Whether the round ends in an answer or in tool
                // calls is only known once it is over, and holding text back until then would silence the reply
                // for the whole generation, so text is forwarded as it arrives.
                var spoken = new StringBuilder();
                // Whitespace before the first visible character is held: a round that only emits blank lines
                // ahead of its tool calls has said nothing.
                StringBuilder? leadingWhitespace = null;
                await foreach (ModelTurnEvent modelEvent in _client
                    .RunAsync(transcript, catalog, context.Options, cancellationToken)
                    .ConfigureAwait(false))
                {
                    switch (modelEvent)
                    {
                        case ThinkingDelta thinking:
                            yield return new TurnThinkingDelta(thinking.Text);
                            break;
                        case ModelTurnToolCall call:
                            calls.Add(call.Call);
                            break;
                        case VisibleTextDelta { Text.Length: > 0 } delta when spoken.Length == 0 && string.IsNullOrWhiteSpace(delta.Text):
                            (leadingWhitespace ??= new StringBuilder()).Append(delta.Text);
                            break;
                        case VisibleTextDelta { Text.Length: > 0 } delta:
                            string text = leadingWhitespace is null ? delta.Text : leadingWhitespace.Append(delta.Text).ToString();
                            leadingWhitespace = null;
                            spoken.Append(text);
                            yield return new TurnTextDelta(Separate(text, ref separatorPending));
                            break;
                        case VisibleTextDelta:
                            break;
                        default:
                            throw new InvalidOperationException($"Unsupported model turn event '{modelEvent.GetType().Name}'.");
                    }
                }

                if (calls.Count == 0)
                {
                    if (leadingWhitespace is not null)
                        yield return new TurnTextDelta(leadingWhitespace.ToString());
                    PublishToolTranscript(context.TurnId, toolTranscript);
                    yield return new TurnFinished();
                    yield break;
                }

                totalToolCalls += calls.Count;
                if (totalToolCalls > MaxToolCallsPerTurn)
                    throw new InvalidOperationException("The model exceeded the per-turn tool call limit.");

                // A provider tool-call batch must be resolved atomically. If it mixes background and
                // foreground preferences, foreground wins so every tool result can be returned in
                // the same provider transcript without leaving dangling call ids.
                bool runInBackground = calls.All(static call => call.ExecutionPreference == ConversationToolExecutionPreference.Background);
                string? executionPrompt = FindCustomBackgroundExecutionPrompt(calls);
                // The next request repeats what the model already said in this round, so it continues from there
                // instead of saying it again. The persisted exchange keeps the calls only: spoken text is
                // committed once, as part of the turn's answer.
                ChatMessage callMessage = CreateToolCallMessage(calls);
                transcript.Add(spoken.Length == 0 ? callMessage : CreateToolCallMessage(calls, spoken.ToString()));
                if (spoken.Length > 0)
                    separatorPending = !char.IsWhiteSpace(spoken[^1]);

                if (runInBackground)
                {
                    var commitGate = new TurnCommitGate();
                    pendingGate = commitGate;
                    ScheduleBackground(context, calls, transcript.ToArray(), catalog, blocklist, commitGate, toolTranscript);
                    try
                    {
                        // The turn ends here, so only a custom prompt needs its own speech boundary.
                        yield return new TurnTextDelta(
                            Separate(executionPrompt ?? CreateBackgroundStartedAnnouncement(calls), ref separatorPending),
                            EndsTextSegment: executionPrompt is not null);

                        foreach (ToolCall call in calls)
                            _history.RecordToolActivity(context.TurnId, call.CallId, call.Tool.Id, ConversationToolResultKind.Accepted);
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

                // The provider batch remains foreground-atomic when preferences are mixed, but a locally
                // configured background prompt is still announced before any tool code starts. Its forced
                // boundary submits even a short or unpunctuated prompt to the consumer's speech pipeline
                // before this iterator is resumed. Model text gets no forced boundary: the consumer would hold
                // the tools back until that text had been synthesized.
                if (executionPrompt is not null)
                {
                    yield return new TurnTextDelta(Separate(executionPrompt, ref separatorPending), EndsTextSegment: true);
                    separatorPending = !char.IsWhiteSpace(executionPrompt[^1]);
                }

                ToolOutcome[] outcomes = await _batches
                    .ExecuteAsync(context.TurnId, calls, blocklist, cancellationToken)
                    .ConfigureAwait(false);
                toolTranscript.Add(callMessage);
                for (int index = 0; index < calls.Count; index++)
                {
                    ChatMessage resultMessage = CreateToolResultMessage(calls[index], outcomes[index]);
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

    private void ScheduleBackground(
        TurnExecutionContext context,
        IReadOnlyList<ToolCall> calls,
        ChatMessage[] transcript,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist blocklist,
        TurnCommitGate commitGate,
        List<ChatMessage> toolTranscript)
    {
        CancellationToken token = _runCancellation?.Token
            ?? throw new InvalidOperationException("The tool conversation session is not started.");
        var registered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_taskGate)
        {
            Task precedingPublication = _backgroundPublicationTail;
            Task task = Task.Run(
                async () =>
                {
                    await registered.Task.ConfigureAwait(false);
                    try
                    {
                        await RunBackgroundAsync(
                            context,
                            calls,
                            transcript,
                            catalog,
                            blocklist,
                            commitGate,
                            precedingPublication,
                            toolTranscript,
                            token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (token.IsCancellationRequested)
                    {
                    }
                    catch (Exception exception)
                    {
                        _logger.LogError(
                            exception,
                            "Background tool work for request {RequestId} could not publish its result.",
                            context.RequestId);
                    }
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
        IReadOnlyList<ToolCall> calls,
        ChatMessage[] initialTranscript,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist blocklist,
        TurnCommitGate commitGate,
        Task precedingPublication,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        if (!await commitGate.WaitAsync().WaitAsync(cancellationToken).ConfigureAwait(false))
            return;

        var allCalls = new List<ToolCall>(calls);
        var outcomes = new Dictionary<string, ToolOutcome>(StringComparer.Ordinal);
        string text = string.Empty;
        try
        {
            text = await RunBackgroundLoopAsync(
                context,
                calls,
                initialTranscript.ToList(),
                catalog,
                blocklist,
                allCalls,
                outcomes,
                toolTranscript,
                cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return;
        }
        catch (Exception exception)
        {
            // Tool work that already finished may have had side effects. A later failure must not rewrite
            // that outcome, so the fixed text below reports every call by what actually happened to it.
            _logger.LogError(
                exception,
                "Background tool work for request {RequestId} failed after {Finished} of {Requested} call(s); reporting what was reached.",
                context.RequestId, outcomes.Count, allCalls.Count);
        }

        if (string.IsNullOrWhiteSpace(text))
            text = CreateBackgroundOutcomeAnnouncement(allCalls, outcomes);

        // Commit the structured exchange and summary together before publishing the result.
        _history.RecordBackgroundResult(context.TurnId, text, toolTranscript);
        if (cancellationToken.IsCancellationRequested)
            return;

        // Submission-order FIFO: execution may overlap, but a later result never jumps ahead of
        // an earlier background acknowledgement in the speech queue.
        try { await precedingPublication.ConfigureAwait(false); } catch { }
        cancellationToken.ThrowIfCancellationRequested();
        await _backgroundOutput.PublishAsync(
            new ConversationBackgroundMessage(Guid.NewGuid(), context.ConversationId, context.RequestId, text),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Runs the accepted batch and lets the model continue until it answers. Nothing is streamed: the answer
    /// is published as one message, so text that accompanies further tool calls is never heard and is dropped.
    /// </summary>
    private async Task<string> RunBackgroundLoopAsync(
        TurnExecutionContext context,
        IReadOnlyList<ToolCall> accepted,
        List<ChatMessage> transcript,
        ToolCatalogSnapshot catalog,
        UnknownOutcomeBlocklist blocklist,
        List<ToolCall> allCalls,
        Dictionary<string, ToolOutcome> outcomes,
        List<ChatMessage> toolTranscript,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ToolCall> batch = accepted;
        int requestedCalls = 0;
        for (int round = 0; round < MaxModelRounds; round++)
        {
            if (round > 0)
                allCalls.AddRange(batch);
            ToolOutcome[] results = await _batches.ExecuteAsync(context.TurnId, batch, blocklist, cancellationToken).ConfigureAwait(false);
            // Persist a batch only once every call has a result. An approval failure must not
            // leave an unmatched call in the history used by later model requests.
            toolTranscript.Add(CreateToolCallMessage(batch));
            for (int index = 0; index < batch.Count; index++)
            {
                outcomes[batch[index].CallId] = results[index];
                ChatMessage resultMessage = CreateToolResultMessage(batch[index], results[index]);
                transcript.Add(resultMessage);
                toolTranscript.Add(resultMessage);
            }

            var requested = new List<ToolCall>();
            var answer = new StringBuilder();
            await foreach (ModelTurnEvent modelEvent in _client
                .RunAsync(transcript, catalog, context.Options, cancellationToken)
                .ConfigureAwait(false))
            {
                if (modelEvent is VisibleTextDelta delta)
                    answer.Append(delta.Text);
                else if (modelEvent is ModelTurnToolCall call)
                    requested.Add(call.Call);
            }

            if (requested.Count == 0)
                return answer.ToString();

            requestedCalls += requested.Count;
            if (requestedCalls > MaxToolCallsPerTurn)
                throw new InvalidOperationException("The background tool loop exceeded its call limit.");
            transcript.Add(CreateToolCallMessage(requested));
            batch = requested;
        }

        throw new InvalidOperationException("The background model loop exceeded its round limit.");
    }

    private string CreateBackgroundStartedAnnouncement(IEnumerable<ToolCall> calls)
    {
        IEnumerable<string> names = calls.Select(static call => call.Tool.Descriptor.DisplayName).Distinct(StringComparer.Ordinal);
        return Format(_announcements.BackgroundStarted, string.Join(_announcements.NameSeparator, names));
    }

    /// <summary>Fixed text for a background result the model could not summarize, one sentence per call.</summary>
    private string CreateBackgroundOutcomeAnnouncement(
        IReadOnlyList<ToolCall> calls,
        IReadOnlyDictionary<string, ToolOutcome> outcomes)
    {
        var text = new StringBuilder();
        foreach (ToolCall call in calls)
        {
            string template = !outcomes.TryGetValue(call.CallId, out ToolOutcome? outcome)
                ? _announcements.BackgroundFailed
                : outcome.Result.Kind switch
                {
                    ConversationToolResultKind.Success => _announcements.BackgroundSucceeded,
                    ConversationToolResultKind.Denied => _announcements.BackgroundDenied,
                    ConversationToolResultKind.Cancelled => _announcements.BackgroundCancelled,
                    ConversationToolResultKind.UnknownOutcome => _announcements.BackgroundUnknownOutcome,
                    _ => _announcements.BackgroundFailed,
                };
            text.Append(Format(template, call.Tool.Descriptor.DisplayName));
        }
        return text.ToString();
    }

    private static string Format(string template, string name) => string.Format(CultureInfo.InvariantCulture, template, name);

    /// <summary>Prefixes the pending separator, unless the text already starts on its own line.</summary>
    private static string Separate(string text, ref bool separatorPending)
    {
        bool separate = separatorPending && !char.IsWhiteSpace(text[0]);
        separatorPending = false;
        return separate ? SegmentSeparator + text : text;
    }

    private static string? FindCustomBackgroundExecutionPrompt(IEnumerable<ToolCall> calls)
        => calls
            .Where(static call => call.ExecutionPreference == ConversationToolExecutionPreference.Background)
            .Select(static call => call.Tool.Descriptor.ExecutionPrompt)
            .FirstOrDefault(static prompt => !string.IsNullOrWhiteSpace(prompt));

    private static ChatMessage CreateToolCallMessage(IReadOnlyList<ToolCall> calls, string? spokenText = null)
    {
        var content = new List<ChatContent>(calls.Count + 1);
        if (!string.IsNullOrEmpty(spokenText))
            content.Add(new TextContent(spokenText));
        foreach (ToolCall call in calls)
            content.Add(new ToolCallContent(call.CallId, call.Tool.ExposedName, call.ArgumentsJson));
        return new ChatMessage(ChatRole.Assistant, content);
    }

    private static ChatMessage CreateToolResultMessage(ToolCall call, ToolOutcome outcome)
        => ChatMessage.ToolResult(call.CallId, outcome.ModelResultJson, !outcome.Result.IsSuccess);

    private void PublishToolTranscript(long turnId, List<ChatMessage> toolTranscript)
    {
        if (toolTranscript.Count == 0)
            return;

        ChatMessage[] messages = toolTranscript.ToArray();
        toolTranscript.Clear();
        _history.RecordToolExchange(turnId, messages);
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        try
        {
            await StopAsync().ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            _logger.LogWarning(exception, "Stopping conversation tools during disposal failed.");
        }
        await _runtime.DisposeAsync().ConfigureAwait(false);
    }
}
