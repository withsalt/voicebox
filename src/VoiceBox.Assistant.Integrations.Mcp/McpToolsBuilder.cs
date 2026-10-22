using VoiceBox.Assistant.Api.Tools.Contracts;

namespace VoiceBox.Assistant.Integrations.Mcp;

public sealed class McpToolsBuilder
{
    private readonly List<IAssistantTool> _tools = [];
    private readonly List<McpServerRegistration> _servers = [];
    private IAssistantToolApprovalPolicy? _approvalPolicy;

    /// <summary>
    /// Registers a local tool whose lifetime is owned by the built VoiceAssistant. The assistant
    /// disposes it after pending calls finish, including when it was never started.
    /// </summary>
    public McpToolsBuilder AddTool(IAssistantTool tool)
    {
        ArgumentNullException.ThrowIfNull(tool);
        _tools.Add(tool);
        return this;
    }

    public McpToolsBuilder AddStdioServer(
        string alias,
        Action<StdioMcpServerOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StdioMcpServerOptions();
        configure(options);
        _servers.Add(options.Freeze(alias));
        return this;
    }

    public McpToolsBuilder AddStreamableHttpServer(
        string alias,
        Action<StreamableHttpMcpServerOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StreamableHttpMcpServerOptions();
        configure(options);
        _servers.Add(options.Freeze(alias));
        return this;
    }

    public McpToolsBuilder UseApprovalPolicy(IAssistantToolApprovalPolicy approvalPolicy)
    {
        ArgumentNullException.ThrowIfNull(approvalPolicy);
        if (_approvalPolicy is not null)
            throw new InvalidOperationException("An MCP tool approval policy is already registered.");
        _approvalPolicy = approvalPolicy;
        return this;
    }

    internal McpToolRuntimeFactory BuildFactory()
    {
        if (_tools.Count == 0 && _servers.Count == 0)
            throw new InvalidOperationException("UseMcpTools must register at least one tool or server.");

        string? duplicateAlias = _servers.GroupBy(static server => server.Alias, StringComparer.Ordinal)
            .FirstOrDefault(static group => group.Count() > 1)?.Key;
        if (duplicateAlias is not null)
            throw new InvalidOperationException($"MCP server alias '{duplicateAlias}' is already registered.");

        string? duplicateLocal = _tools.GroupBy(static tool => tool.Descriptor.Name, StringComparer.Ordinal)
            .FirstOrDefault(static group => group.Count() > 1)?.Key;
        if (duplicateLocal is not null)
            throw new InvalidOperationException($"Local tool '{duplicateLocal}' is already registered.");

        return new McpToolRuntimeFactory(_tools.ToArray(), _servers.ToArray(), _approvalPolicy);
    }

    internal static void ValidateAlias(string alias)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(alias);
        if (alias.Length > 64 || alias.Any(static character =>
            !char.IsAsciiLetterOrDigit(character) && character is not '_' and not '-'))
        {
            throw new ArgumentException(
                "Server alias must be 1-64 ASCII letters, digits, underscores, or hyphens.",
                nameof(alias));
        }
    }
}
