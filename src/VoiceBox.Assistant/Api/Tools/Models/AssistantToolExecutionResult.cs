namespace VoiceBox.Assistant.Api.Tools.Models;

/// <summary>进程内工具返回给模型的规范化结果。</summary>
public sealed class AssistantToolExecutionResult
{
    private AssistantToolExecutionResult(
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

    public static AssistantToolExecutionResult Success(
        string content,
        string? structuredContentJson = null) =>
        new(true, content, structuredContentJson, null);

    public static AssistantToolExecutionResult Error(
        string errorCode,
        string content) =>
        new(false, content, null, errorCode);
}
