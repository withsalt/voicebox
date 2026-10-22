using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceBox.Assistant.Integrations.Mcp;

public abstract class McpToolServerOptions
{
    public ISet<string> AllowedTools { get; } = new HashSet<string>(StringComparer.Ordinal);

    public TimeSpan ConnectTimeout { get; set; } = TimeSpan.FromSeconds(3);

    public TimeSpan RequestTimeout { get; set; } = TimeSpan.FromSeconds(15);

    public int MaxResultBytes { get; set; } = 1024 * 1024;

    public int MaxArgumentBytes { get; set; } = 256 * 1024;

    public int MaxConcurrentCalls { get; set; } = 4;

    public AssistantToolRisk DefaultRisk { get; set; } = AssistantToolRisk.Sensitive;

    public AssistantToolExecutionPolicy ExecutionPolicy { get; set; } =
        AssistantToolExecutionPolicy.ModelChoice;

    /// <summary>Use the MCP Tasks extension and poll task-backed tool calls to completion.</summary>
    public bool UseMcpTasks { get; set; }

    public int TaskPollingIntervalMs { get; set; } = 1000;

    /// <summary>Trusted host-controlled prefix used for acknowledgement speech.</summary>
    public string DisplayName { get; set; } = string.Empty;

    internal abstract McpServerRegistration Freeze(string alias);

    protected void ValidateCommon(string alias)
    {
        McpToolsBuilder.ValidateAlias(alias);
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
}
