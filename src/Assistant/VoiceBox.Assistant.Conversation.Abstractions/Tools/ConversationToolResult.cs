namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 一次工具调用的规范化结果。本地工具和工具来源返回的是同一类型。
/// </summary>
public sealed class ConversationToolResult
{
    private ConversationToolResult(
        ConversationToolResultKind kind,
        string? content,
        string? structuredContentJson,
        string? errorCode,
        bool isTransient)
    {
        Kind = kind;
        Content = content ?? string.Empty;
        StructuredContentJson = string.IsNullOrWhiteSpace(structuredContentJson) ? null : structuredContentJson;
        ErrorCode = errorCode;
        IsTransient = isTransient;
    }

    /// <summary>规范化结果分类。</summary>
    public ConversationToolResultKind Kind { get; }

    /// <summary>调用是否成功完成。</summary>
    public bool IsSuccess => Kind == ConversationToolResultKind.Success;

    /// <summary>纯文本结果或错误说明; 没有时为空字符串。</summary>
    public string Content { get; }

    /// <summary>结构化结果, 必须是合法 JSON; 没有时为 null。</summary>
    public string? StructuredContentJson { get; }

    /// <summary>宿主可识别的错误码; 成功时为 null。</summary>
    public string? ErrorCode { get; }

    /// <summary>
    /// 失败是否为瞬时故障。为 true 时, 消费方可以对没有副作用的工具重试同一次调用; 成功结果恒为 false。
    /// </summary>
    public bool IsTransient { get; }

    /// <summary>成功结果。</summary>
    public static ConversationToolResult Success(string content, string? structuredContentJson = null)
        => new(ConversationToolResultKind.Success, content, structuredContentJson, null, false);

    /// <summary>工具自身报告的失败, 分类为 <see cref="ConversationToolResultKind.ToolError"/>。</summary>
    public static ConversationToolResult Error(string errorCode, string content, bool isTransient = false)
        => Failure(ConversationToolResultKind.ToolError, errorCode, content, isTransient: isTransient);

    /// <summary>指定分类的非成功结果, 例如超时、协议错误或结果未知。</summary>
    public static ConversationToolResult Failure(
        ConversationToolResultKind kind,
        string errorCode,
        string? content = null,
        string? structuredContentJson = null,
        bool isTransient = false)
    {
        if (!Enum.IsDefined(kind) || kind == ConversationToolResultKind.Success)
            throw new ArgumentOutOfRangeException(nameof(kind));
        ArgumentException.ThrowIfNullOrWhiteSpace(errorCode);
        return new ConversationToolResult(kind, content, structuredContentJson, errorCode, isTransient);
    }
}
