using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using ModelContextProtocol;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Extensions.Tasks;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Mcp;

/// <summary>
/// 一个 MCP 工具来源。它拥有 client 连接, 并把协议结果转换为公共工具结果模型。
/// 审批、Schema 校验、并发/重试预算与历史关联由消费该来源的会话负责。
/// </summary>
/// <remarks>
/// 使用 <see cref="CreateStdio"/> 或 <see cref="CreateStreamableHttp"/> 创建实例。创建只校验并
/// 冻结配置, 不启动进程、不连接网络、不发现工具; 这些动作发生在 <see cref="StartAsync"/>。
/// 一个实例对应一个 MCP Server。
/// </remarks>
public sealed class McpToolProvider : IConversationToolProvider, IConversationToolProviderMetadata
{
    private readonly McpServerRegistration[] _servers;
    private readonly Dictionary<ConversationToolId, RegisteredTool> _tools = [];
    private readonly List<McpClient> _clients = [];
    private readonly List<SemaphoreSlim> _serverGates = [];
    private IReadOnlyList<ConversationToolDefinition> _catalog = Array.Empty<ConversationToolDefinition>();
    private bool _started;
    private int _disposed;

    private McpToolProvider(IReadOnlyList<McpServerRegistration> servers)
    {
        ArgumentNullException.ThrowIfNull(servers);
        _servers = servers.ToArray();
        if (_servers.Length == 0)
            throw new ArgumentException("At least one MCP server registration is required.", nameof(servers));
        if (_servers.Select(static server => server.Alias).Distinct(StringComparer.Ordinal).Count() != _servers.Length)
        {
            throw new ArgumentException("MCP server aliases must be unique.", nameof(servers));
        }
        ServerAliases = Array.AsReadOnly(_servers.Select(static server => server.Alias).ToArray());
    }

    /// <summary>创建 stdio MCP 工具来源。</summary>
    public static McpToolProvider CreateStdio(string alias, Action<StdioMcpServerOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StdioMcpServerOptions();
        configure(options);
        return new McpToolProvider([options.Freeze(alias)]);
    }

    /// <summary>创建 Streamable HTTP MCP 工具来源。</summary>
    public static McpToolProvider CreateStreamableHttp(string alias, Action<StreamableHttpMcpServerOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StreamableHttpMcpServerOptions();
        configure(options);
        return new McpToolProvider([options.Freeze(alias)]);
    }

    /// <summary>本来源在连接前即可确定的 server 别名。</summary>
    public IReadOnlyList<string> ServerAliases { get; }

    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_started)
            throw new InvalidOperationException("The MCP tool provider is already started.");

        _tools.Clear();
        try
        {
            foreach (McpServerRegistration registration in _servers)
                await AddServerAsync(registration, cancellationToken).ConfigureAwait(false);

            _catalog = Array.AsReadOnly(_tools.Values.Select(static value => value.Definition).ToArray());
            _started = true;
        }
        catch
        {
            await DisposeClientsAsync().ConfigureAwait(false);
            DisposeServerGates();
            _tools.Clear();
            throw;
        }
    }

    public ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (!_started)
            throw new InvalidOperationException("The MCP tool provider is not started.");
        return ValueTask.FromResult(_catalog);
    }

    public async ValueTask<ConversationToolResult> InvokeAsync(ConversationToolCall call, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_started)
            throw new InvalidOperationException("The MCP tool provider is not started.");
        if (!_tools.TryGetValue(call.ToolId, out RegisteredTool? registered))
        {
            return new ConversationToolResult(call.CallId, ConversationToolResultKind.ProtocolError, ErrorCode: "unknown_mcp_tool");
        }

        return await ExecuteMcpAsync(registered, call, cancellationToken).ConfigureAwait(false);
    }

    public Task StopAsync()
    {
        _started = false;
        _catalog = Array.Empty<ConversationToolDefinition>();
        _tools.Clear();
        return StopCoreAsync();
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        _started = false;
        _catalog = Array.Empty<ConversationToolDefinition>();
        _tools.Clear();
        await StopCoreAsync().ConfigureAwait(false);
    }

    private async Task StopCoreAsync()
    {
        await DisposeClientsAsync().ConfigureAwait(false);
        DisposeServerGates();
    }

    private async Task AddServerAsync(McpServerRegistration registration, CancellationToken cancellationToken)
    {
        IClientTransport transport = CreateTransport(registration);
        using var connect = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        connect.CancelAfter(registration.ConnectTimeout);
        McpClient client;
        try
        {
            client = await McpClient.CreateAsync(
                transport,
                clientOptions: null,
                NullLoggerFactory.Instance,
                connect.Token).ConfigureAwait(false);
        }
        catch
        {
            if (transport is IAsyncDisposable disposable)
                await disposable.DisposeAsync().ConfigureAwait(false);
            throw;
        }

        _clients.Add(client);
        var executionGate = new SemaphoreSlim(registration.MaxConcurrentCalls, registration.MaxConcurrentCalls);
        _serverGates.Add(executionGate);
        using var discovery = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        discovery.CancelAfter(registration.RequestTimeout);
        IList<McpClientTool> discovered = await client.ListToolsAsync(options: null, discovery.Token).ConfigureAwait(false);
        foreach (McpClientTool mcpTool in discovered)
        {
            if (registration.AllowedTools.Count > 0 && !registration.AllowedTools.Contains(mcpTool.Name))
            {
                continue;
            }

            Tool protocol = mcpTool.ProtocolTool;
            var descriptor = new ConversationToolDescriptor(
                protocol.Name,
                registration.DisplayName,
                protocol.Description ?? string.Empty,
                protocol.InputSchema.GetRawText(),
                protocol.OutputSchema?.GetRawText(),
                registration.DefaultRisk,
                registration.ExecutionPolicy,
                executionPrompt: null,
                timeout: registration.RequestTimeout);
            var definition = new ConversationToolDefinition(new ConversationToolId(registration.Alias, protocol.Name), descriptor);
            if (!_tools.TryAdd(definition.Id, new RegisteredTool(definition, mcpTool, client, registration, executionGate)))
            {
                throw new InvalidOperationException($"Duplicate MCP tool identity '{definition.Id}'.");
            }
        }
    }

    private static IClientTransport CreateTransport(McpServerRegistration registration)
    {
        if (registration.Transport == McpServerTransport.Stdio)
        {
            Dictionary<string, string?> environment = StdioClientTransportOptions.GetDefaultEnvironmentVariables();
            foreach ((string name, string? value) in registration.EnvironmentVariables)
                environment[name] = value;
            return new StdioClientTransport(new StdioClientTransportOptions
            {
                Name = registration.Alias,
                Command = registration.Command!,
                Arguments = registration.Arguments.ToArray(),
                EnvironmentVariables = environment,
                InheritEnvironmentVariables = false,
                WorkingDirectory = registration.WorkingDirectory,
            }, NullLoggerFactory.Instance);
        }

        return new HttpClientTransport(new HttpClientTransportOptions
        {
            Name = registration.Alias,
            Endpoint = registration.Endpoint!,
            TransportMode = HttpTransportMode.StreamableHttp,
            ConnectionTimeout = registration.ConnectTimeout,
            AdditionalHeaders = new Dictionary<string, string>(registration.Headers),
        }, NullLoggerFactory.Instance);
    }

    private static async ValueTask<ConversationToolResult> ExecuteMcpAsync(
        RegisteredTool registered,
        ConversationToolCall call,
        CancellationToken cancellationToken)
    {
        McpClientTool tool = registered.McpTool ?? throw new InvalidOperationException("MCP tool registration has no client tool.");
        McpServerRegistration server = registered.Server
            ?? throw new InvalidOperationException("MCP tool registration has no server options.");
        if (Encoding.UTF8.GetByteCount(call.ArgumentsJson) > server.MaxArgumentBytes)
        {
            return new ConversationToolResult(call.CallId, ConversationToolResultKind.ProtocolError, ErrorCode: "arguments_too_large");
        }

        using JsonDocument document = JsonDocument.Parse(string.IsNullOrWhiteSpace(call.ArgumentsJson) ? "{}" : call.ArgumentsJson);
        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (JsonProperty property in document.RootElement.EnumerateObject())
            arguments.Add(property.Name, property.Value.Clone());

        SemaphoreSlim executionGate = registered.ExecutionGate
            ?? throw new InvalidOperationException("MCP tool registration has no execution gate.");
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(server.RequestTimeout);
        bool acquired = false;
        bool dispatched = false;
        CallToolResult result;
        try
        {
            await executionGate.WaitAsync(timeout.Token).ConfigureAwait(false);
            acquired = true;
            timeout.Token.ThrowIfCancellationRequested();
            if (server.UseMcpTasks)
            {
                McpClient client = registered.McpClient ?? throw new InvalidOperationException("MCP task tool registration has no client.");
                var taskArguments = arguments.ToDictionary(
                    static item => item.Key,
                    static item => (JsonElement)item.Value!,
                    StringComparer.Ordinal);
                dispatched = true;
                result = await client.CallToolWithPollingAsync(
                    new CallToolRequestParams
                    {
                        Name = tool.ProtocolTool.Name,
                        Arguments = taskArguments,
                    },
                    server.TaskPollingIntervalMs,
                    timeout.Token).ConfigureAwait(false);
            }
            else
            {
                dispatched = true;
                result = await tool.CallAsync(arguments, progress: null, options: null, timeout.Token).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (
            timeout.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            // After dispatch, a timeout cannot prove that a remote mutation did not happen.
            bool unknownOutcome = dispatched && server.DefaultRisk is
                ConversationToolRisk.SideEffecting or
                ConversationToolRisk.Destructive or
                ConversationToolRisk.Privileged;
            return new ConversationToolResult(
                call.CallId,
                unknownOutcome ? ConversationToolResultKind.UnknownOutcome : ConversationToolResultKind.TimedOut,
                ErrorCode: "timeout");
        }
        finally
        {
            if (acquired)
                executionGate.Release();
        }
        if (TryDescribeUnsupportedContent(result) is { } unsupported)
        {
            return new ConversationToolResult(
                call.CallId,
                ConversationToolResultKind.UnsupportedCapability,
                unsupported,
                "unsupported_content_type");
        }

        string resultJson = NormalizeMcpResult(result);
        if (Encoding.UTF8.GetByteCount(resultJson) > server.MaxResultBytes)
        {
            return new ConversationToolResult(call.CallId, ConversationToolResultKind.ProtocolError, ErrorCode: "result_too_large");
        }

        return new ConversationToolResult(
            call.CallId,
            result.IsError == true
                ? ConversationToolResultKind.ToolError
                : ConversationToolResultKind.Success,
            resultJson,
            result.IsError == true ? "mcp_tool_error" : null);
    }

    /// <summary>
    /// Reports a result this release cannot represent for the model. Structured content and text
    /// are supported, so only a result whose payload is exclusively non-text blocks is rejected.
    /// </summary>
    private static string? TryDescribeUnsupportedContent(CallToolResult result)
    {
        if (result.StructuredContent is JsonElement)
            return null;
        if (result.Content.Count == 0 || result.Content.OfType<TextContentBlock>().Any())
            return null;

        string[] kinds = result.Content
            .Select(static block => block.Type)
            .Where(static type => !string.IsNullOrWhiteSpace(type))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        return JsonSerializer.Serialize(new
        {
            error = "unsupported_content_type",
            message = "This tool returned content this host cannot consume. Only text and structured content are supported.",
            contentTypes = kinds,
        });
    }

    private static string NormalizeMcpResult(CallToolResult result)
    {
        if (result.StructuredContent is JsonElement structured)
            return structured.GetRawText();

        string[] text = result.Content.OfType<TextContentBlock>().Select(static block => block.Text).ToArray();
        return JsonSerializer.Serialize(new { content = string.Join("\n", text) });
    }

    private async Task DisposeClientsAsync()
    {
        for (int index = _clients.Count - 1; index >= 0; index--)
        {
            try { await _clients[index].DisposeAsync().ConfigureAwait(false); } catch { }
        }
        _clients.Clear();
    }

    private void DisposeServerGates()
    {
        foreach (SemaphoreSlim gate in _serverGates)
            gate.Dispose();
        _serverGates.Clear();
    }

    private sealed record RegisteredTool(
        ConversationToolDefinition Definition,
        McpClientTool? McpTool,
        McpClient? McpClient,
        McpServerRegistration? Server,
        SemaphoreSlim? ExecutionGate);
}
