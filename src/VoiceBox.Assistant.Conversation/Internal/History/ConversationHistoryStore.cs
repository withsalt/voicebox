using System.Text;
using System.Text.Json;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.History;

/// <summary>
/// Owns one session's model transcript and the pending/committed state of each turn. Only the
/// session execution flow drives it; no external port can read or mutate a half-built turn.
/// </summary>
internal sealed class ConversationHistoryStore
{
    private const string AdditionalContextSeparator = "\n\n[附加上下文]\n";

    private readonly object _lock = new();
    private readonly string _systemPrompt;
    private readonly List<ChatMessage> _history = new();
    private readonly List<ConversationTurnRecord> _turnRecords = new();
    private readonly Dictionary<long, ChatMessage> _turnUserMessages = new();
    private readonly HashSet<long> _committedResponseTurnIds = new();
    private readonly int _maxHistoryTurns;
    private readonly int _toolTranscriptRetentionTurns;
    private readonly int _maxPersistedToolResultBytes;

    public ConversationHistoryStore(
        string systemPrompt,
        int maxHistoryTurns,
        int toolTranscriptRetentionTurns = 0,
        int maxPersistedToolResultBytes = 8 * 1024)
    {
        if (maxHistoryTurns < 1)
            throw new ArgumentOutOfRangeException(nameof(maxHistoryTurns), "Max history turns must be >= 1.");
        if (toolTranscriptRetentionTurns < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(toolTranscriptRetentionTurns),
                "Tool transcript retention turns must be >= 0.");
        }
        if (maxPersistedToolResultBytes < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxPersistedToolResultBytes),
                "Max persisted tool result bytes must be >= 1.");
        }

        _systemPrompt = systemPrompt ?? string.Empty;
        _maxHistoryTurns = maxHistoryTurns;
        _toolTranscriptRetentionTurns = toolTranscriptRetentionTurns;
        _maxPersistedToolResultBytes = maxPersistedToolResultBytes;
        _history.Add(ChatMessage.System(_systemPrompt));
    }

    /// <summary>
    /// Appends the user message and returns the prompt for this turn. Additional context is only
    /// attached to the returned system message; the committed transcript keeps the base prompt and
    /// never turns retrieved context into a permanent user message.
    /// </summary>
    public ChatMessage AddUserMessage(long turnId, string userText, string? additionalContext, out ChatMessage[] prompt)
    {
        var userMessage = ChatMessage.User(userText);
        lock (_lock)
        {
            _history.Add(userMessage);
            _turnUserMessages[turnId] = userMessage;
            _turnRecords.Add(new ConversationTurnRecord(
                turnId,
                userMessage,
                AssistantMessage: null,
                Array.Empty<ToolCall>(),
                Array.Empty<ToolResult>(),
                Array.Empty<ChatMessage>()));
            prompt = BuildPromptLocked(additionalContext);
        }

        return userMessage;
    }

    private ChatMessage[] BuildPromptLocked(string? additionalContext)
    {
        ChatMessage[] prompt = _history.ToArray();
        if (!string.IsNullOrWhiteSpace(additionalContext))
        {
            prompt[0] = ChatMessage.System(
                _systemPrompt.Length == 0
                    ? additionalContext.Trim()
                    : _systemPrompt + AdditionalContextSeparator + additionalContext.Trim());
        }

        return prompt;
    }

    /// <summary>
    /// Commits a completely generated turn: tool messages go between the user message and the
    /// spoken answer, then the assistant text (when non-empty). Empty or thinking-only replies
    /// still commit the user message and are trimmed by the same rules.
    /// </summary>
    public bool CommitGeneratedResponse(long turnId, string assistantText)
    {
        lock (_lock)
        {
            if (!_turnUserMessages.TryGetValue(turnId, out ChatMessage? userMessage))
                return false;
            return CommitGeneratedResponseLocked(turnId, userMessage, assistantText);
        }
    }

    private bool CommitGeneratedResponseLocked(
        long turnId,
        ChatMessage userMessage,
        string assistantText)
    {
        if (!_committedResponseTurnIds.Add(turnId))
            return false;

        if (_history.FindIndex(m => ReferenceEquals(m, userMessage)) == -1)
            _history.Add(userMessage);

        int turnRecordIndex = _turnRecords.FindIndex(record =>
            record.TurnId == turnId && ReferenceEquals(record.UserMessage, userMessage));
        if (turnRecordIndex >= 0)
            AppendToolMessagesLocked(_turnRecords[turnRecordIndex].ToolMessages);

        ChatMessage? assistantMessage = null;
        if (!string.IsNullOrEmpty(assistantText))
        {
            assistantMessage = ChatMessage.Assistant(assistantText);
            _history.Add(assistantMessage);
        }
        if (turnRecordIndex == -1)
        {
            _turnRecords.Add(new ConversationTurnRecord(
                turnId,
                userMessage,
                assistantMessage,
                Array.Empty<ToolCall>(),
                Array.Empty<ToolResult>(),
                Array.Empty<ChatMessage>()));
        }
        else
        {
            _turnRecords[turnRecordIndex] = _turnRecords[turnRecordIndex] with
            {
                AssistantMessage = assistantMessage,
            };
        }
        _turnUserMessages.Remove(turnId);

        // Even an empty assistant reply commits the user message, so trimming must run
        // unconditionally to keep a long-running session bounded.
        TrimHistoryLocked();
        return true;
    }

    /// <summary>
    /// Removes an unanswered turn. Every exit path that did not commit the turn calls this;
    /// committed turns are never rolled back by a later enumerator disposal.
    /// </summary>
    public void RemoveUnansweredUserMessage(long turnId)
    {
        lock (_lock)
        {
            if (_committedResponseTurnIds.Contains(turnId))
                return;

            if (!_turnUserMessages.TryGetValue(turnId, out ChatMessage? userMessage))
                return;

            _history.RemoveAll(message => ReferenceEquals(message, userMessage));
            _turnUserMessages.Remove(turnId);
            RemovePendingTurnRecordLocked(turnId, userMessage);
        }
    }

    private void AppendToolMessagesLocked(IReadOnlyList<ChatMessage> toolMessages)
    {
        foreach (ChatMessage message in toolMessages)
        {
            if (!_history.Any(value => ReferenceEquals(value, message)))
                _history.Add(message);
        }
    }

    public void RecordToolActivity(
        long turnId,
        ToolCall call,
        ToolResult result)
    {
        ArgumentNullException.ThrowIfNull(call);
        ArgumentNullException.ThrowIfNull(result);
        lock (_lock)
        {
            int index = _turnRecords.FindLastIndex(record => record.TurnId == turnId);
            if (index < 0)
                return;
            UpsertToolCallAndResultLocked(index, call, result);
        }
    }

    /// <summary>
    /// Stores the provider transcript slice a tool round produced. The messages stay pending on
    /// the turn record until the turn commits, so an interrupted turn discards them together with
    /// its user message and no orphaned tool call can reach a later prompt.
    /// </summary>
    public void RecordToolExchange(long turnId, IReadOnlyList<ChatMessage> toolMessages)
    {
        ArgumentNullException.ThrowIfNull(toolMessages);
        if (_toolTranscriptRetentionTurns == 0 || toolMessages.Count == 0)
            return;

        lock (_lock)
        {
            int index = _turnRecords.FindLastIndex(record => record.TurnId == turnId);
            if (index < 0)
                return;

            ConversationTurnRecord record = _turnRecords[index];
            var retained = new List<ChatMessage>(record.ToolMessages);
            foreach (ChatMessage message in toolMessages)
                retained.Add(TruncateToolResultIfNeeded(message));
            _turnRecords[index] = record with { ToolMessages = retained.ToArray() };
        }
    }

    private ChatMessage TruncateToolResultIfNeeded(ChatMessage message)
    {
        if (message.Role != ChatRole.Tool ||
            message.Content.Count != 1 ||
            message.Content[0] is not ToolResultContent result ||
            Encoding.UTF8.GetByteCount(result.ResultJson) <= _maxPersistedToolResultBytes)
        {
            return message;
        }

        string replacement = JsonSerializer.Serialize(new
        {
            __truncated = true,
            __originalBytes = Encoding.UTF8.GetByteCount(result.ResultJson),
            note = "This tool result was truncated for context retention. Call the tool again if the full payload is required.",
        });
        return ChatMessage.ToolResult(result.CallId, replacement, result.IsError);
    }

    /// <summary>
    /// Adds the background tool exchange and summary atomically to a committed originating turn.
    /// Background execution is admitted only after the foreground acknowledgement commits.
    /// Results for canceled or expired turns do not recreate those turns.
    /// </summary>
    public bool RecordBackgroundResult(
        long turnId,
        string assistantText,
        IReadOnlyList<ChatMessage> toolMessages)
    {
        ArgumentNullException.ThrowIfNull(toolMessages);
        if (string.IsNullOrWhiteSpace(assistantText))
            return false;

        lock (_lock)
        {
            int index = _turnRecords.FindLastIndex(record => record.TurnId == turnId);
            if (index < 0 || !_committedResponseTurnIds.Contains(turnId))
                return false;

            ConversationTurnRecord record = _turnRecords[index];
            ChatMessage[] retained = _toolTranscriptRetentionTurns == 0
                ? []
                : toolMessages.Select(TruncateToolResultIfNeeded).ToArray();
            var message = ChatMessage.Assistant(assistantText);
            _turnRecords[index] = record with
            {
                ToolMessages = record.ToolMessages.Concat(retained).ToArray(),
                BackgroundAssistantMessages = record.BackgroundAssistantMessages
                    .Append(message)
                    .ToArray(),
            };

            AppendToolMessagesLocked(retained);
            _history.Add(message);
            TrimHistoryLocked();
            return true;
        }
    }

    internal int CommittedResponseTurnCount
    {
        get
        {
            lock (_lock)
                return _committedResponseTurnIds.Count;
        }
    }

    /// <summary>
    /// Committed model transcript copy (including the system message). Used internally to build
    /// the next prompt; callers outside the session get a copied projection.
    /// </summary>
    public ChatMessage[] GetModelTranscriptSnapshot()
    {
        lock (_lock)
            return _history.ToArray();
    }

    /// <summary>
    /// Committed model transcript without the system message, copied for an external observer.
    /// </summary>
    public IReadOnlyList<ChatMessage> GetCommittedTranscriptCopy()
    {
        ChatMessage[] snapshot;
        lock (_lock)
            snapshot = _history.Skip(1).ToArray();
        return Array.AsReadOnly(snapshot.Select(CopyMessage).ToArray());
    }

    /// <summary>
    /// Product-facing projection: the user turns and the assistant replies they saw.
    /// </summary>
    public ChatMessage[] GetUserConversationHistorySnapshot()
    {
        ChatMessage[] snapshot;
        lock (_lock)
            snapshot = _history.Skip(1).Where(IsUserVisible).ToArray();
        return snapshot.Select(CopyMessage).ToArray();
    }

    private static ChatMessage CopyMessage(ChatMessage message)
        => new(message.Role, message.Content.Select(CopyContent).ToArray());

    private static ChatContent CopyContent(ChatContent content) => content switch
    {
        TextContent text => new TextContent(text.Text),
        ToolCallContent call => new ToolCallContent(call.CallId, call.ToolName, call.ArgumentsJson),
        ToolResultContent result => new ToolResultContent(result.CallId, result.ResultJson, result.IsError),
        AudioContent audio when audio.Bytes is not null => new AudioContent(audio.Bytes),
        AudioContent audio when audio.Path is not null => new AudioContent(audio.Path),
        ImageContent image when image.Bytes is not null => new ImageContent(image.Bytes),
        ImageContent image when image.Path is not null => new ImageContent(image.Path),
        _ => throw new InvalidOperationException(
            $"Unsupported chat content type '{content.GetType().Name}'."),
    };

    private static bool IsUserVisible(ChatMessage message)
    {
        if (message.Role == ChatRole.Tool)
            return false;
        return !message.Content.Any(static content => content is ToolCallContent);
    }

    /// <summary>
    /// Audit projection over per-turn records. A result contributes one entry when its call ID is
    /// correlated in the same turn, including outcomes recorded before a model round completes.
    /// </summary>
    public ToolAuditRecord[] GetToolAuditTrailSnapshot()
    {
        lock (_lock)
        {
            var audit = new List<ToolAuditRecord>();
            foreach (ConversationTurnRecord record in _turnRecords)
            {
                foreach (ToolResult result in record.ToolResults)
                {
                    ToolCall? call = record.ToolCalls.FirstOrDefault(
                        value => string.Equals(value.CallId, result.CallId, StringComparison.Ordinal));
                    if (call is null)
                        continue;

                    audit.Add(new ToolAuditRecord(
                        record.TurnId,
                        result.CallId,
                        call.ToolId,
                        result.Kind));
                }
            }

            return audit.ToArray();
        }
    }

    /// <summary>
    /// Keeps the most recent <c>MaxHistoryTurns</c> user/assistant turns; the system message stays
    /// pinned. Text-only turns trim by message count so their prompt stays byte-identical; once a
    /// retained turn carries tool correlation, whole turns are dropped instead.
    /// </summary>
    private void TrimHistoryLocked()
    {
        if (HasToolCorrelatedRecordsLocked())
        {
            TrimToolTranscriptRetentionLocked();
            TrimToolCorrelatedTurnsLocked();
            return;
        }

        long maxConvMessages = (long)_maxHistoryTurns * 2;
        int convMessages = _history.Count - 1;
        if (convMessages <= maxConvMessages) return;

        int drop = checked((int)(convMessages - maxConvMessages));
        _history.RemoveRange(1, drop);

        if (_history[1].Role != ChatRole.User)
            _history.RemoveAt(1);

        _turnRecords.RemoveAll(record =>
            !_history.Any(message => ReferenceEquals(message, record.UserMessage)));
        TrimCommittedResponseTurnIdsLocked();
    }

    private bool HasToolCorrelatedRecordsLocked()
    {
        foreach (ConversationTurnRecord record in _turnRecords)
        {
            if (record.ToolCalls.Count > 0 ||
                record.ToolResults.Count > 0 ||
                record.ToolMessages.Count > 0)
            {
                return true;
            }
        }

        return false;
    }

    private void TrimToolCorrelatedTurnsLocked()
    {
        while (_turnRecords.Count > _maxHistoryTurns)
        {
            ConversationTurnRecord oldest = _turnRecords[0];
            _turnRecords.RemoveAt(0);
            _committedResponseTurnIds.Remove(oldest.TurnId);
            _history.RemoveAll(message =>
                ReferenceEquals(message, oldest.UserMessage) ||
                (oldest.AssistantMessage is not null &&
                    ReferenceEquals(message, oldest.AssistantMessage)) ||
                oldest.BackgroundAssistantMessages.Any(background =>
                    ReferenceEquals(message, background)) ||
                oldest.ToolMessages.Any(tool => ReferenceEquals(message, tool)));
        }
    }

    private void TrimToolTranscriptRetentionLocked()
    {
        int turnsWithToolMessages = 0;
        for (int index = 0; index < _turnRecords.Count; index++)
        {
            if (_turnRecords[index].ToolMessages.Count > 0)
                turnsWithToolMessages++;
        }
        if (turnsWithToolMessages <= _toolTranscriptRetentionTurns)
            return;

        int dropCount = turnsWithToolMessages - _toolTranscriptRetentionTurns;
        for (int index = 0; index < _turnRecords.Count && dropCount > 0; index++)
        {
            ConversationTurnRecord record = _turnRecords[index];
            if (record.ToolMessages.Count == 0)
                continue;

            _history.RemoveAll(message =>
                record.ToolMessages.Any(tool => ReferenceEquals(message, tool)));
            _turnRecords[index] = record with { ToolMessages = Array.Empty<ChatMessage>() };
            dropCount--;
        }
    }

    private void RemovePendingTurnRecordLocked(long turnId, ChatMessage userMessage)
    {
        for (int index = _turnRecords.Count - 1; index >= 0; index--)
        {
            ConversationTurnRecord record = _turnRecords[index];
            if (record.TurnId != turnId)
                continue;
            if (record.AssistantMessage is not null)
                continue;
            if (!ReferenceEquals(record.UserMessage, userMessage))
                continue;

            if (record.ToolMessages.Count > 0)
            {
                record = record with { ToolMessages = Array.Empty<ChatMessage>() };
                _turnRecords[index] = record;
            }

            if (record.ToolCalls.Count > 0 ||
                record.ToolResults.Count > 0 ||
                record.BackgroundAssistantMessages.Count > 0)
            {
                continue;
            }
            _turnRecords.RemoveAt(index);
        }
    }

    private void UpsertToolCallAndResultLocked(
        int recordIndex,
        ToolCall call,
        ToolResult result)
    {
        UpsertToolCallLocked(recordIndex, call);
        UpsertToolResultLocked(recordIndex, result);
    }

    private void UpsertToolCallLocked(int recordIndex, ToolCall call)
    {
        ConversationTurnRecord record = _turnRecords[recordIndex];
        var calls = record.ToolCalls.ToList();
        int existing = calls.FindIndex(value =>
            string.Equals(value.CallId, call.CallId, StringComparison.Ordinal));
        if (existing < 0)
            calls.Add(call);
        else
            calls[existing] = call;
        _turnRecords[recordIndex] = record with { ToolCalls = calls.ToArray() };
    }

    private void UpsertToolResultLocked(int recordIndex, ToolResult result)
    {
        ConversationTurnRecord record = _turnRecords[recordIndex];
        var results = record.ToolResults.ToList();
        int existing = results.FindIndex(value =>
            string.Equals(value.CallId, result.CallId, StringComparison.Ordinal));
        if (existing < 0)
        {
            results.Add(result);
        }
        else if (result.Kind != ConversationToolResultKind.Accepted ||
            results[existing].Kind == ConversationToolResultKind.Accepted)
        {
            // A delayed acknowledgement must never overwrite the terminal outcome of a fast
            // background execution.
            results[existing] = result;
        }
        _turnRecords[recordIndex] = record with { ToolResults = results.ToArray() };
    }

    private void TrimCommittedResponseTurnIdsLocked()
    {
        _committedResponseTurnIds.RemoveWhere(turnId =>
            !_turnRecords.Exists(record => record.TurnId == turnId));
    }
}
