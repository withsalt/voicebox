using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Mcp;

public abstract class McpToolServerOptions
{
    public ISet<string> AllowedTools { get; } = new HashSet<string>(StringComparer.Ordinal);

    public TimeSpan ConnectTimeout { get; set; } = TimeSpan.FromSeconds(3);

    /// <summary>工具发现及每次调用的超时。调用超时包含等待并发名额、请求执行和 MCP Tasks 轮询。</summary>
    public TimeSpan RequestTimeout { get; set; } = TimeSpan.FromSeconds(15);

    public int MaxResultBytes { get; set; } = 1024 * 1024;

    public int MaxArgumentBytes { get; set; } = 256 * 1024;

    public int MaxConcurrentCalls { get; set; } = 4;

    public ConversationToolRisk DefaultRisk { get; set; } = ConversationToolRisk.Sensitive;

    public ConversationToolExecutionPolicy ExecutionPolicy { get; set; } = ConversationToolExecutionPolicy.ModelChoice;

    /// <summary>Use the MCP Tasks extension and poll task-backed tool calls to completion.</summary>
    public bool UseMcpTasks { get; set; }

    public int TaskPollingIntervalMs { get; set; } = 1000;

    /// <summary>Trusted host-controlled prefix used for acknowledgement speech.</summary>
    public string DisplayName { get; set; } = string.Empty;

    internal abstract McpServerRegistration Freeze(string alias);

    protected void ValidateCommon(string alias)
    {
        ValidateAlias(alias);
        if (ConnectTimeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(ConnectTimeout));
        if (RequestTimeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(RequestTimeout));
        if (MaxResultBytes < 1024)
            throw new ArgumentOutOfRangeException(nameof(MaxResultBytes));
        if (MaxArgumentBytes < 1024)
            throw new ArgumentOutOfRangeException(nameof(MaxArgumentBytes));
        if (MaxConcurrentCalls < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxConcurrentCalls));
        if (!Enum.IsDefined(DefaultRisk))
            throw new ArgumentOutOfRangeException(nameof(DefaultRisk));
        if (!Enum.IsDefined(ExecutionPolicy))
            throw new ArgumentOutOfRangeException(nameof(ExecutionPolicy));
        if (TaskPollingIntervalMs < 100)
            throw new ArgumentOutOfRangeException(nameof(TaskPollingIntervalMs));
    }

    private static void ValidateAlias(string alias)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(alias);
        if (alias.Length > 64 || alias.Any(static character => !char.IsAsciiLetterOrDigit(character) && character is not '_' and not '-'))
        {
            throw new ArgumentException("Server alias must be 1-64 ASCII letters, digits, underscores, or hyphens.", nameof(alias));
        }
        if (string.Equals(alias, "local", StringComparison.Ordinal))
        {
            throw new ArgumentException("MCP server alias 'local' is reserved for AddTool registrations.", nameof(alias));
        }
    }
}
