namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具执行返回给消费方的规范化结果。</summary>
public sealed class ConversationToolExecutionResult
{
    private ConversationToolExecutionResult(
        bool isSuccess,
        string content,
        string? structuredContentJson,
        string? errorCode)
    {
        IsSuccess = isSuccess;
        Content = content ?? string.Empty;
        StructuredContentJson = structuredContentJson;
        ErrorCode = errorCode;
    }

    public bool IsSuccess { get; }

    public string Content { get; }

    public string? StructuredContentJson { get; }

    public string? ErrorCode { get; }

    public static ConversationToolExecutionResult Success(
        string content,
        string? structuredContentJson = null) =>
        new(true, content, structuredContentJson, null);

    public static ConversationToolExecutionResult Error(
        string errorCode,
        string content) =>
        new(false, content, null, errorCode);
}
