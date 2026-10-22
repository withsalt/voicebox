namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

using VoiceBox.Assistant.Api.Tools.Models;

internal sealed record AssistantToolCall
{
    public AssistantToolCall(
        string callId,
        AssistantToolId toolId,
        OwnedJsonValue arguments,
        AssistantToolExecutionPreference executionPreference = AssistantToolExecutionPreference.Foreground)
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

    public AssistantToolId ToolId { get; }

    public OwnedJsonValue Arguments { get; }

    public string ArgumentsHash { get; }

    public AssistantToolExecutionPreference ExecutionPreference { get; }
}
