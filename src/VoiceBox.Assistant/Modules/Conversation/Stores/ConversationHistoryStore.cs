using System.Text;
using System.Text.Json;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Conversation.Models;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Conversation.Stores;

/// <summary>
/// 维护 LLM prompt 历史，并支持一轮回复的提交/回滚语义。
/// </summary>
internal sealed class ConversationHistoryStore :
    IToolHistorySink,
    IConversationRollbackPort,
    IConversationPromptPort,
    IConversationBackgroundTextPort
{
    private readonly object _lock = new();
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

        _maxHistoryTurns = maxHistoryTurns;
        _toolTranscriptRetentionTurns = toolTranscriptRetentionTurns;
        _maxPersistedToolResultBytes = maxPersistedToolResultBytes;
        _history.Add(ChatMessage.System(systemPrompt));
    }

    public ChatMessage AddUserMessage(long turnId, string userText, out ChatMessage[] prompt)
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
                Array.Empty<AssistantToolCall>(),
                Array.Empty<AssistantToolResult>(),
                Array.Empty<ChatMessage>()));
            prompt = _history.ToArray();
        }

        return userMessage;
    }

    /// <summary>
    /// LLM 流结束、本轮回复文本完整产出时落库:
    /// 把 assistant 回复接在对应 user 消息之后。提交发生在 TTS 播放排空之前, 因此即便随后
    /// 用户开口 barge-in 打断播放, 这次"已完整生成"的问答也会被保留。
    /// 不校验回合是否仍是当前回合: 已完整生成的回复无论是否随后被打断都应入库。
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

    public bool CommitGeneratedResponse(long turnId, ChatMessage userMessage, string assistantText)
    {
        lock (_lock)
            return CommitGeneratedResponseLocked(turnId, userMessage, assistantText);
    }

    private bool CommitGeneratedResponseLocked(
        long turnId,
        ChatMessage userMessage,
        string assistantText)
    {
        if (!_committedResponseTurnIds.Add(turnId))
            return false;

        // 正常情况下 userMessage 仍是会话末尾 (本轮入栈后未再追加其它消息), 追加 assistant
        // 即等价于紧跟其后。若它已被并发的打断清理逻辑移除 (极少见的竞态), 连同 user 一起补回,
        // 保证这次完整问答不丢失。
        if (_history.FindIndex(m => ReferenceEquals(m, userMessage)) == -1)
            _history.Add(userMessage);

        int turnRecordIndex = _turnRecords.FindIndex(record =>
            record.TurnId == turnId && ReferenceEquals(record.UserMessage, userMessage));
        IReadOnlyList<ChatMessage> backgroundMessages = turnRecordIndex < 0
            ? Array.Empty<ChatMessage>()
            : _turnRecords[turnRecordIndex].BackgroundAssistantMessages;

        // Tool call/result messages belong between the user message and the spoken answer, which is
        // exactly the order the provider observed while producing that answer.
        if (turnRecordIndex >= 0)
            AppendToolMessagesLocked(_turnRecords[turnRecordIndex].ToolMessages);

        ChatMessage? assistantMessage = null;
        if (!string.IsNullOrEmpty(assistantText))
        {
            assistantMessage = ChatMessage.Assistant(assistantText);
            _history.Add(assistantMessage);
        }
        foreach (ChatMessage backgroundMessage in backgroundMessages)
        {
            if (!_history.Any(message => ReferenceEquals(message, backgroundMessage)))
                _history.Add(backgroundMessage);
        }
        if (turnRecordIndex == -1)
        {
            _turnRecords.Add(new ConversationTurnRecord(
                turnId,
                userMessage,
                assistantMessage,
                Array.Empty<AssistantToolCall>(),
                Array.Empty<AssistantToolResult>(),
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

        // 即使 assistantText 为空 (LLM 仅产出 thinking / 流为空), 用户消息已经入栈,
        // 必须无条件裁剪, 否则长时间运行下 _history 会单调增长。
        TrimHistoryLocked();
        return true;
    }

    public void RemoveUnansweredUserMessage(long turnId, ChatMessage? userMessage = null)
    {
        lock (_lock)
        {
            if (_committedResponseTurnIds.Contains(turnId))
                return;

            if (userMessage is null && !_turnUserMessages.TryGetValue(turnId, out userMessage))
                return;

            if (TryCommitDeferredBackgroundResponseLocked(turnId, userMessage))
                return;

            int index = _history.FindIndex(m => ReferenceEquals(m, userMessage));
            if (index == -1)
            {
                _turnUserMessages.Remove(turnId);
                RemovePendingTurnRecordLocked(turnId, userMessage);
                return;
            }

            // A trailing unanswered user is never followed by an assistant message. Handle
            // that case first so the following-role check cannot index past the list.
            if (index == _history.Count - 1)
            {
                _history.RemoveAt(index);
                _turnUserMessages.Remove(turnId);
                RemovePendingTurnRecordLocked(turnId, userMessage);
                return;
            }

            if (_history[index + 1].Role == ChatRole.Assistant)
            {
                _turnUserMessages.Remove(turnId);
                return;
            }

            _history.RemoveAt(index);
            _turnUserMessages.Remove(turnId);
            RemovePendingTurnRecordLocked(turnId, userMessage);
        }
    }

    public void ForgetCommittedResponseTurn(long turnId)
    {
        lock (_lock)
            _committedResponseTurnIds.Remove(turnId);
    }

    /// <summary>
    /// Appends tool messages that are not already present. Identity is by reference, matching how
    /// background assistant messages are reconciled, so a path that runs twice cannot duplicate an
    /// exchange.
    /// </summary>
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
        AssistantToolCall call,
        AssistantToolResult result)
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
    /// Stores the provider transcript slice a tool round produced. The messages stay pending on the
    /// turn record until the turn commits, so an interrupted turn discards them together with its
    /// user message and no orphaned tool call can reach a later prompt.
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

    /// <summary>
    /// Caps a persisted tool result so a single large payload cannot dominate later prompts. The
    /// replacement stays valid JSON and is explicitly marked, so the model can tell the content is
    /// incomplete rather than silently reading a truncated document. Only the persisted copy is
    /// affected; the round that produced the result always saw it in full.
    /// </summary>
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
    /// Adds a deferred assistant result to the originating turn. If the acknowledgement is still
    /// being committed, the message stays pending in the turn record and is inserted immediately
    /// after it; if the foreground response was rolled back, the valid background result restores
    /// the user/result exchange so the next model prompt still observes the completed work.
    /// </summary>
    public bool RecordBackgroundAssistantText(long turnId, string assistantText)
    {
        if (string.IsNullOrWhiteSpace(assistantText))
            return false;

        lock (_lock)
        {
            int index = _turnRecords.FindLastIndex(record => record.TurnId == turnId);
            if (index < 0)
                return false;

            ConversationTurnRecord record = _turnRecords[index];
            var message = ChatMessage.Assistant(assistantText);
            _turnRecords[index] = record with
            {
                BackgroundAssistantMessages = record.BackgroundAssistantMessages
                    .Append(message)
                    .ToArray(),
            };

            bool mainResponseStillPending =
                record.AssistantMessage is null &&
                _turnUserMessages.ContainsKey(turnId) &&
                !_committedResponseTurnIds.Contains(turnId);
            if (mainResponseStillPending)
                return true;

            if (!_history.Any(value => ReferenceEquals(value, record.UserMessage)))
                _history.Add(record.UserMessage);
            _history.Add(message);
            if (record.AssistantMessage is null)
                _committedResponseTurnIds.Add(turnId);
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

    public ChatMessage[] GetModelTranscriptSnapshot()
    {
        lock (_lock)
            return _history.ToArray();
    }

    /// <summary>
    /// Product-facing projection: the user turns and the assistant replies they saw. Tool results
    /// and tool-call messages are model-facing plumbing and never appear here, even though they are
    /// part of the model transcript.
    /// </summary>
    public ChatMessage[] GetUserConversationHistorySnapshot()
    {
        lock (_lock)
            return _history.Skip(1).Where(IsUserVisible).ToArray();
    }

    private static bool IsUserVisible(ChatMessage message)
    {
        if (message.Role == ChatRole.Tool)
            return false;
        // An assistant message carrying tool calls has no spoken text; only its later summary is
        // user-visible.
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
                foreach (AssistantToolResult result in record.ToolResults)
                {
                    // A result is correlated to its call by CallId inside the same turn, so a
                    // trimmed turn can never leave a dangling audit entry.
                    AssistantToolCall? call = record.ToolCalls.FirstOrDefault(
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
    /// 仅保留最近 MaxHistoryTurns 轮 user/assistant 对话，system 消息始终置顶。
    /// </summary>
    /// <remarks>
    /// Trimming stays message-based for text-only turns so the prompt sent to the model is
    /// byte-identical to a composition without tool support. Once a retained turn carries tool
    /// call/result correlation, <see cref="TrimToolCorrelatedTurnsLocked"/> takes over and
    /// drops whole turns so a call and its result can never be split.
    /// </remarks>
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
            // Retained tool messages break the two-messages-per-turn assumption that message-based
            // trimming relies on, so they must also route to the turn-atomic path.
            if (record.ToolCalls.Count > 0 ||
                record.ToolResults.Count > 0 ||
                record.ToolMessages.Count > 0)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Turn-atomic trim: drops the oldest complete turns until the limit is met, removing every
    /// message that belongs to a dropped turn together.
    /// </summary>
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

    /// <summary>
    /// Context-budget trim: keeps tool call/result messages only for the most recent retention
    /// window. Older turns degrade to plain user/assistant text, so the conversation thread stays
    /// intact while structured tool payloads stop accumulating.
    /// </summary>
    /// <remarks>
    /// A turn's exchange is dropped as a whole, so a retained call can never lose its result. The
    /// window counts turns that carry a tool exchange, not all turns, because interleaved text-only
    /// turns should not push tool context out of scope.
    /// </remarks>
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

            // A discarded turn's tool messages were never woven into the transcript and never will
            // be, so they must stop occupying a retention slot a committed turn could use. Tool
            // call/result correlation stays, because the audit trail records work that really ran.
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

    private bool TryCommitDeferredBackgroundResponseLocked(long turnId, ChatMessage userMessage)
    {
        int index = _turnRecords.FindLastIndex(record =>
            record.TurnId == turnId && ReferenceEquals(record.UserMessage, userMessage));
        if (index < 0)
            return false;

        ConversationTurnRecord record = _turnRecords[index];
        if (record.AssistantMessage is not null || record.BackgroundAssistantMessages.Count == 0)
            return false;

        if (!_history.Any(message => ReferenceEquals(message, userMessage)))
            _history.Add(userMessage);
        // The tool exchange that produced this background result precedes its summary, so a turn
        // whose foreground answer was rolled back still yields a well-ordered transcript.
        AppendToolMessagesLocked(record.ToolMessages);
        foreach (ChatMessage backgroundMessage in record.BackgroundAssistantMessages)
        {
            if (!_history.Any(message => ReferenceEquals(message, backgroundMessage)))
                _history.Add(backgroundMessage);
        }
        _committedResponseTurnIds.Add(turnId);
        _turnUserMessages.Remove(turnId);
        TrimHistoryLocked();
        return true;
    }

    private void UpsertToolCallAndResultLocked(
        int recordIndex,
        AssistantToolCall call,
        AssistantToolResult result)
    {
        UpsertToolCallLocked(recordIndex, call);
        UpsertToolResultLocked(recordIndex, result);
    }

    private void UpsertToolCallLocked(int recordIndex, AssistantToolCall call)
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

    private void UpsertToolResultLocked(int recordIndex, AssistantToolResult result)
    {
        ConversationTurnRecord record = _turnRecords[recordIndex];
        var results = record.ToolResults.ToList();
        int existing = results.FindIndex(value =>
            string.Equals(value.CallId, result.CallId, StringComparison.Ordinal));
        if (existing < 0)
        {
            results.Add(result);
        }
        else if (result.Kind != AssistantToolResultKind.Accepted ||
            results[existing].Kind == AssistantToolResultKind.Accepted)
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
