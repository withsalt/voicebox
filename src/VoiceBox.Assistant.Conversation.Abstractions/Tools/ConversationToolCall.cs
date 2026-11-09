namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 工具来源收到的一次规范化调用。参数为 JSON 文本, 由来源自行解析。
/// </summary>
public sealed record ConversationToolCall(
    string CallId,
    ConversationToolId ToolId,
    string ArgumentsJson,
    ConversationToolExecutionPreference ExecutionPreference = ConversationToolExecutionPreference.Foreground)
{
    public string CallId { get; } = string.IsNullOrWhiteSpace(CallId)
        ? throw new ArgumentException("Tool call id cannot be empty.", nameof(CallId))
        : CallId;

    public ConversationToolId ToolId { get; } =
        string.IsNullOrWhiteSpace(ToolId.ServerAlias) || string.IsNullOrWhiteSpace(ToolId.ToolName)
            ? throw new ArgumentException(
                "Tool identity must contain a server alias and tool name.", nameof(ToolId))
            : ToolId;

    public string ArgumentsJson { get; } = ArgumentsJson
        ?? throw new ArgumentNullException(nameof(ArgumentsJson));

    public ConversationToolExecutionPreference ExecutionPreference { get; } =
        Enum.IsDefined(ExecutionPreference)
            ? ExecutionPreference
            : throw new ArgumentOutOfRangeException(nameof(ExecutionPreference));
}
