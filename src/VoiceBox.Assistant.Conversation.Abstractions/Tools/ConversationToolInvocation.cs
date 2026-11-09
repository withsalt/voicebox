using System.Text.Json;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>本地工具收到的一次调用。</summary>
public sealed class ConversationToolInvocation
{
    /// <summary>
    /// 创建一个调用输入。
    /// </summary>
    /// <param name="callId">调用关联 ID, 不得为空。</param>
    /// <param name="arguments">JSON 对象参数。构造时复制, 调用方可以随后释放原始 <see cref="JsonDocument"/>。</param>
    /// <param name="executionPreference">本次调用的前后台执行偏好。</param>
    public ConversationToolInvocation(string callId, JsonElement arguments, ConversationToolExecutionPreference executionPreference)
    {
        if (string.IsNullOrWhiteSpace(callId))
            throw new ArgumentException("Tool call id cannot be empty.", nameof(callId));
        if (arguments.ValueKind != JsonValueKind.Object)
            throw new ArgumentException("Tool arguments must be a JSON object.", nameof(arguments));
        if (!Enum.IsDefined(executionPreference))
            throw new ArgumentOutOfRangeException(nameof(executionPreference));

        CallId = callId;
        Arguments = arguments.Clone();
        ExecutionPreference = executionPreference;
    }

    public string CallId { get; }

    public JsonElement Arguments { get; }

    public ConversationToolExecutionPreference ExecutionPreference { get; }
}
