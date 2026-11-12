namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>审批策略看到的调用上下文。</summary>
/// <param name="Tool">被调用工具的受信任元数据。</param>
/// <param name="Call">即将执行的调用; 工具来源别名见 <see cref="ConversationToolCall.ToolId"/>。</param>
/// <param name="ArgumentsHash">参数规范化文本的摘要, 用于日志关联; 不是抗碰撞摘要, 不能作为授权缓存键。</param>
public sealed record ConversationToolApprovalContext(ConversationToolDescriptor Tool, ConversationToolCall Call, string ArgumentsHash)
{
    public ConversationToolDescriptor Tool { get; } = Tool ?? throw new ArgumentNullException(nameof(Tool));

    public ConversationToolCall Call { get; } = Call ?? throw new ArgumentNullException(nameof(Call));

    public string ArgumentsHash { get; } = string.IsNullOrWhiteSpace(ArgumentsHash)
        ? throw new ArgumentException("Arguments hash cannot be empty.", nameof(ArgumentsHash))
        : ArgumentsHash;
}
