namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 结构化工具调用的运行参数。
/// </summary>
public sealed class ConversationToolOptions
{
    /// <summary>
    /// 最近多少轮对话在模型 prompt 中保留 tool call / tool result 消息。更早的轮次退化为
    /// 纯 user/assistant 文本。取 0 表示完全不持久化工具消息。
    /// </summary>
    public int ToolTranscriptRetentionTurns { get; set; } = 3;

    /// <summary>
    /// 单条 tool result 落库时的字节上限。超出的结果会被截断并标注。
    /// 仅影响持久化副本; 产生该结果的那一轮模型续推始终看到完整结果。
    /// </summary>
    public int MaxPersistedToolResultBytes { get; set; } = 8 * 1024;

    /// <summary>
    /// 同一批 provider tool call 的最大并发执行数。只有 ReadOnly / Sensitive 风险的工具参与并发。
    /// </summary>
    public int MaxParallelToolCalls { get; set; } = 4;

    /// <summary>只读工具瞬时失败的最大重试次数。</summary>
    public int MaxToolRetryAttempts { get; set; } = 2;

    /// <summary>首次重试前的退避时长, 之后每次乘 3。</summary>
    public int ToolRetryBaseDelayMs { get; set; } = 200;

    /// <summary>单个工具调用含全部重试的总耗时上限。</summary>
    public int MaxToolRetryBudgetMs { get; set; } = 60_000;

    /// <summary>是否在审批前按工具的 input JSON Schema 校验参数。参数始终必须是 JSON 对象。</summary>
    public bool ValidateToolArguments { get; set; } = true;

    internal ConversationToolOptions Clone()
    {
        var clone = new ConversationToolOptions
        {
            ToolTranscriptRetentionTurns = ToolTranscriptRetentionTurns,
            MaxPersistedToolResultBytes = MaxPersistedToolResultBytes,
            MaxParallelToolCalls = MaxParallelToolCalls,
            MaxToolRetryAttempts = MaxToolRetryAttempts,
            ToolRetryBaseDelayMs = ToolRetryBaseDelayMs,
            MaxToolRetryBudgetMs = MaxToolRetryBudgetMs,
            ValidateToolArguments = ValidateToolArguments,
        };
        Validate(clone);
        return clone;
    }

    internal static void Validate(ConversationToolOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.ToolTranscriptRetentionTurns < 0)
            throw new ArgumentOutOfRangeException(
                nameof(options.ToolTranscriptRetentionTurns),
                "ToolTranscriptRetentionTurns must be >= 0.");
        if (options.MaxPersistedToolResultBytes < 1)
            throw new ArgumentOutOfRangeException(nameof(options.MaxPersistedToolResultBytes), "MaxPersistedToolResultBytes must be >= 1.");
        if (options.MaxParallelToolCalls < 1)
            throw new ArgumentOutOfRangeException(nameof(options.MaxParallelToolCalls), "MaxParallelToolCalls must be >= 1.");
        if (options.MaxToolRetryAttempts < 0)
            throw new ArgumentOutOfRangeException(nameof(options.MaxToolRetryAttempts), "MaxToolRetryAttempts must be >= 0.");
        if (options.ToolRetryBaseDelayMs < 0)
            throw new ArgumentOutOfRangeException(nameof(options.ToolRetryBaseDelayMs), "ToolRetryBaseDelayMs must be >= 0.");
        if (options.MaxToolRetryBudgetMs < 0)
            throw new ArgumentOutOfRangeException(nameof(options.MaxToolRetryBudgetMs), "MaxToolRetryBudgetMs must be >= 0.");
    }
}
