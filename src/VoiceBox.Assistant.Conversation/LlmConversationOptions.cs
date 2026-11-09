namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// <see cref="LlmConversationSession"/> 的配置。
/// </summary>
public sealed class LlmConversationOptions
{
    /// <summary>
    /// 会话的 system 提示词。默认空字符串; 语音助手等场景由调用方显式配置自己的提示词。
    /// </summary>
    public string SystemPrompt { get; set; } = string.Empty;

    /// <summary>
    /// 最近保留多少轮 user/assistant 对话 (system 消息始终置顶, 不计入)。
    /// </summary>
    public int MaxHistoryTurns { get; set; } = 20;

    /// <summary>
    /// 单次模型推理的 token 上限。null 表示沿用 provider 默认值。
    /// </summary>
    public int? MaxResponseTokens { get; set; }

    /// <summary>
    /// 是否请求模型输出思考。思考通过 <c>ConversationThinkingDelta</c> 对外可观察,
    /// 但不进入正式历史或回答文字。
    /// </summary>
    public bool EnableThinking { get; set; }

    internal LlmConversationOptions Clone()
    {
        var clone = new LlmConversationOptions
        {
            SystemPrompt = SystemPrompt ?? string.Empty,
            MaxHistoryTurns = MaxHistoryTurns,
            MaxResponseTokens = MaxResponseTokens,
            EnableThinking = EnableThinking,
        };
        Validate(clone);
        return clone;
    }

    internal static void Validate(LlmConversationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.MaxHistoryTurns < 1)
            throw new ArgumentOutOfRangeException(
                nameof(options.MaxHistoryTurns),
                "MaxHistoryTurns must be >= 1.");
        if (options.MaxResponseTokens is <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(options.MaxResponseTokens),
                "MaxResponseTokens must be positive when specified.");
    }
}
