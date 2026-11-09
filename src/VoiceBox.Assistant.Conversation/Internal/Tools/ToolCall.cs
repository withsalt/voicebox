using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

internal sealed record ToolCall
{
    public ToolCall(
        string callId,
        ConversationToolId toolId,
        OwnedJsonValue arguments,
        ConversationToolExecutionPreference executionPreference = ConversationToolExecutionPreference.Foreground)
    {
        if (string.IsNullOrWhiteSpace(callId))
            throw new ArgumentException("Tool call id cannot be empty.", nameof(callId));
        ArgumentNullException.ThrowIfNull(arguments);
        if (string.IsNullOrWhiteSpace(toolId.ServerAlias) || string.IsNullOrWhiteSpace(toolId.ToolName))
            throw new ArgumentException("Tool identity must contain a server alias and tool name.", nameof(toolId));
        CallId = callId;
        ToolId = toolId;
        Arguments = arguments;
        ArgumentsHash = arguments.CanonicalHash;
        ExecutionPreference = executionPreference;
    }

    public string CallId { get; }

    public ConversationToolId ToolId { get; }

    public OwnedJsonValue Arguments { get; }

    public string ArgumentsHash { get; }

    public ConversationToolExecutionPreference ExecutionPreference { get; }
}
