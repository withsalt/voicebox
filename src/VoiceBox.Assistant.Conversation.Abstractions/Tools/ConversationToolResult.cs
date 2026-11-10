namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 工具来源返回的一次规范化结果。
/// </summary>
/// <param name="CallId">对应的调用 ID。</param>
/// <param name="Kind">规范化结果分类。</param>
/// <param name="ValueJson">结构化结果 JSON; 无结果时为 null。</param>
/// <param name="ErrorCode">宿主可识别的错误码; 成功时为 null。</param>
public sealed record ConversationToolResult(
    string CallId,
    ConversationToolResultKind Kind,
    string? ValueJson = null,
    string? ErrorCode = null)
{
    public string CallId { get; } = string.IsNullOrWhiteSpace(CallId)
        ? throw new ArgumentException("Tool result call id cannot be empty.", nameof(CallId))
        : CallId;

    public ConversationToolResultKind Kind { get; } = Enum.IsDefined(Kind) ? Kind : throw new ArgumentOutOfRangeException(nameof(Kind));

    public string? ValueJson { get; } = ValueJson;

    public string? ErrorCode { get; } = ErrorCode;
}
