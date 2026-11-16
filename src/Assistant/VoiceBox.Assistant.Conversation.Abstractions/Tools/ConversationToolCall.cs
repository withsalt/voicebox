using System.Text.Json;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 一次工具调用。本地工具、工具来源和审批策略收到的是同一类型。
/// </summary>
public sealed class ConversationToolCall
{
    /// <summary>
    /// 创建一次调用。
    /// </summary>
    /// <param name="callId">调用关联 ID, 不得为空。</param>
    /// <param name="toolId">被调用工具的身份。</param>
    /// <param name="arguments">JSON 对象参数。构造时复制, 调用方可以随后释放原始 <see cref="JsonDocument"/>。</param>
    /// <param name="executionPreference">本次调用的前后台执行偏好。</param>
    public ConversationToolCall(
        string callId,
        ConversationToolId toolId,
        JsonElement arguments,
        ConversationToolExecutionPreference executionPreference = ConversationToolExecutionPreference.Foreground)
    {
        if (string.IsNullOrWhiteSpace(callId))
            throw new ArgumentException("Tool call id cannot be empty.", nameof(callId));
        if (string.IsNullOrWhiteSpace(toolId.ServerAlias) || string.IsNullOrWhiteSpace(toolId.ToolName))
            throw new ArgumentException("Tool identity must contain a server alias and tool name.", nameof(toolId));
        if (arguments.ValueKind != JsonValueKind.Object)
            throw new ArgumentException("Tool arguments must be a JSON object.", nameof(arguments));
        if (!Enum.IsDefined(executionPreference))
            throw new ArgumentOutOfRangeException(nameof(executionPreference));

        CallId = callId;
        ToolId = toolId;
        Arguments = arguments.Clone();
        ExecutionPreference = executionPreference;
    }

    /// <summary>
    /// 从 JSON 文本创建一次调用。空白文本视为空对象; 不是合法 JSON 对象时抛出 <see cref="ArgumentException"/>。
    /// </summary>
    public ConversationToolCall(
        string callId,
        ConversationToolId toolId,
        string argumentsJson,
        ConversationToolExecutionPreference executionPreference = ConversationToolExecutionPreference.Foreground)
        : this(callId, toolId, ParseArguments(argumentsJson), executionPreference)
    {
    }

    /// <summary>调用关联 ID。</summary>
    public string CallId { get; }

    /// <summary>被调用工具的身份。</summary>
    public ConversationToolId ToolId { get; }

    /// <summary>JSON 对象参数。实例独立持有, 不依赖创建它的 <see cref="JsonDocument"/>。</summary>
    public JsonElement Arguments { get; }

    /// <summary>本次调用的前后台执行偏好。</summary>
    public ConversationToolExecutionPreference ExecutionPreference { get; }

    private static JsonElement ParseArguments(string argumentsJson)
    {
        ArgumentNullException.ThrowIfNull(argumentsJson);
        try
        {
            using JsonDocument document = JsonDocument.Parse(string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson);
            return document.RootElement.Clone();
        }
        catch (JsonException exception)
        {
            throw new ArgumentException("Tool arguments are not valid JSON.", nameof(argumentsJson), exception);
        }
    }
}
