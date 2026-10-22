namespace VoiceBox.Assistant.Integrations.Mcp;

public sealed class StdioMcpServerOptions : McpToolServerOptions
{
    public string Command { get; set; } = string.Empty;

    public IList<string> Arguments { get; } = new List<string>();

    public IDictionary<string, string?> EnvironmentVariables { get; } =
        new Dictionary<string, string?>(StringComparer.Ordinal);

    public string? WorkingDirectory { get; set; }

    internal override McpServerRegistration Freeze(string alias)
    {
        ValidateCommon(alias);
        ArgumentException.ThrowIfNullOrWhiteSpace(Command);
        return McpServerRegistration.Stdio(
            alias,
            Command,
            Arguments.ToArray(),
            new Dictionary<string, string?>(EnvironmentVariables, StringComparer.Ordinal),
            WorkingDirectory,
            this);
    }
}
