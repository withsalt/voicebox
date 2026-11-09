using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

internal sealed record ToolResult
{
    public ToolResult(string callId, ConversationToolResultKind kind, OwnedJsonValue? value, string? errorCode = null)
    {
        if (string.IsNullOrWhiteSpace(callId))
            throw new ArgumentException("Tool result call id cannot be empty.", nameof(callId));
        CallId = callId;
        Kind = kind;
        Value = value;
        ErrorCode = errorCode;
    }

    public string CallId { get; }

    public ConversationToolResultKind Kind { get; }

    public OwnedJsonValue? Value { get; }

    public string? ErrorCode { get; }

    /// <summary>Model-facing JSON for the provider transcript; never emitted directly to TTS.</summary>
    public string ToModelResultJson()
    {
        if (Value is not null)
            return Value.CanonicalJson;

        string error = System.Text.Json.JsonSerializer.Serialize(ErrorCode ?? Kind.ToString());
        string outcome = System.Text.Json.JsonSerializer.Serialize(Kind.ToString());
        return $"{{\"error\":{error},\"outcome\":{outcome}}}";
    }

    public static ToolResult From(ConversationToolResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        return new ToolResult(
            result.CallId,
            result.Kind,
            result.ValueJson is null ? null : OwnedJsonValue.Parse(result.ValueJson),
            result.ErrorCode);
    }
}
