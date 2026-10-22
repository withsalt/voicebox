using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceBox.Assistant.Integrations.Mcp;

internal sealed record McpServerRegistration(
    string Alias,
    McpServerTransport Transport,
    string? Command,
    IReadOnlyList<string> Arguments,
    IReadOnlyDictionary<string, string?> EnvironmentVariables,
    string? WorkingDirectory,
    Uri? Endpoint,
    IReadOnlyDictionary<string, string> Headers,
    IReadOnlySet<string> AllowedTools,
    TimeSpan ConnectTimeout,
    TimeSpan RequestTimeout,
    int MaxResultBytes,
    int MaxArgumentBytes,
    int MaxConcurrentCalls,
    AssistantToolRisk DefaultRisk,
    AssistantToolExecutionPolicy ExecutionPolicy,
    string DisplayName,
    bool UseMcpTasks,
    int TaskPollingIntervalMs)
{
    public static McpServerRegistration Stdio(
        string alias,
        string command,
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string?> environment,
        string? workingDirectory,
        McpToolServerOptions options) =>
        Create(alias, McpServerTransport.Stdio, command, arguments, environment,
            workingDirectory, null, new Dictionary<string, string>(), options);

    public static McpServerRegistration StreamableHttp(
        string alias,
        Uri endpoint,
        IReadOnlyDictionary<string, string> headers,
        McpToolServerOptions options) =>
        Create(alias, McpServerTransport.StreamableHttp, null, [],
            new Dictionary<string, string?>(), null, endpoint, headers, options);

    private static McpServerRegistration Create(
        string alias,
        McpServerTransport transport,
        string? command,
        IReadOnlyList<string> arguments,
        IReadOnlyDictionary<string, string?> environment,
        string? workingDirectory,
        Uri? endpoint,
        IReadOnlyDictionary<string, string> headers,
        McpToolServerOptions options) => new(
            alias,
            transport,
            command,
            arguments,
            environment,
            workingDirectory,
            endpoint,
            headers,
            new HashSet<string>(options.AllowedTools, StringComparer.Ordinal),
            options.ConnectTimeout,
            options.RequestTimeout,
            options.MaxResultBytes,
            options.MaxArgumentBytes,
            options.MaxConcurrentCalls,
            options.DefaultRisk,
            options.ExecutionPolicy,
            string.IsNullOrWhiteSpace(options.DisplayName) ? alias : options.DisplayName,
            options.UseMcpTasks,
            options.TaskPollingIntervalMs);
}
