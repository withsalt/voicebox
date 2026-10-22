namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal readonly record struct AssistantToolId
{
    public AssistantToolId(string serverAlias, string toolName)
    {
        if (string.IsNullOrWhiteSpace(serverAlias))
            throw new ArgumentException("Server alias cannot be empty.", nameof(serverAlias));
        if (string.IsNullOrWhiteSpace(toolName))
            throw new ArgumentException("Tool name cannot be empty.", nameof(toolName));
        ServerAlias = serverAlias;
        ToolName = toolName;
    }

    public string ServerAlias { get; }

    public string ToolName { get; }
}
