namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具在其来源内的稳定身份。</summary>
public readonly record struct ConversationToolId
{
    public ConversationToolId(string serverAlias, string toolName)
    {
        if (string.IsNullOrWhiteSpace(serverAlias))
            throw new ArgumentException("Server alias cannot be empty.", nameof(serverAlias));
        if (string.IsNullOrWhiteSpace(toolName))
            throw new ArgumentException("Tool name cannot be empty.", nameof(toolName));
        ServerAlias = serverAlias;
        ToolName = toolName;
    }

    /// <summary>工具来源的稳定别名, 用于在同一会话内隔离不同来源。</summary>
    public string ServerAlias { get; }

    /// <summary>来源内的工具名。</summary>
    public string ToolName { get; }

    public override string ToString() => $"{ServerAlias}/{ToolName}";
}
