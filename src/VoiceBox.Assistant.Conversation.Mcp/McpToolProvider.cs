using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
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
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger _logger;
    private readonly Dictionary<ConversationToolId, RegisteredTool> _tools = [];
    private readonly List<McpClient> _clients = [];
    private readonly List<SemaphoreSlim> _serverGates = [];
    private IReadOnlyList<ConversationToolDefinition> _catalog = Array.Empty<ConversationToolDefinition>();
    private bool _started;
    private int _disposed;

    private McpToolProvider(IReadOnlyList<McpServerRegistration> servers, ILoggerFactory? loggerFactory)
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
        _loggerFactory = loggerFactory ?? NullLoggerFactory.Instance;
        _logger = _loggerFactory.CreateLogger<McpToolProvider>();
    }

    /// <summary>创建 stdio MCP 工具来源。</summary>
    /// <param name="alias">工具来源别名。</param>
    /// <param name="configure">配置 server 选项; 返回后配置即被冻结。</param>
    /// <param name="loggerFactory">可选。接收 MCP client、transport 与本来源的诊断日志; 未提供时不输出日志。</param>
    public static McpToolProvider CreateStdio(string alias, Action<StdioMcpServerOptions> configure, ILoggerFactory? loggerFactory = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StdioMcpServerOptions();
        configure(options);
        return new McpToolProvider([options.Freeze(alias)], loggerFactory);
    }

    /// <summary>创建 Streamable HTTP MCP 工具来源。</summary>
    /// <param name="alias">工具来源别名。</param>
    /// <param name="configure">配置 server 选项; 返回后配置即被冻结。</param>
    /// <param name="loggerFactory">可选。接收 MCP client、transport 与本来源的诊断日志; 未提供时不输出日志。</param>
    public static McpToolProvider CreateStreamableHttp(
        string alias,
        Action<StreamableHttpMcpServerOptions> configure,
        ILoggerFactory? loggerFactory = null)
    {
        ArgumentNullException.ThrowIfNull(configure);
        var options = new StreamableHttpMcpServerOptions();
        configure(options);
        return new McpToolProvider([options.Freeze(alias)], loggerFactory);
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
        ArgumentNullException.ThrowIfNull(call);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_started)
            throw new InvalidOperationException("The MCP tool provider is not started.");
        if (!_tools.TryGetValue(call.ToolId, out RegisteredTool? registered))
            return ConversationToolResult.Failure(ConversationToolResultKind.ProtocolError, "unknown_mcp_tool");

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
            client = await McpClient.CreateAsync(transport, clientOptions: null, _loggerFactory, connect.Token).ConfigureAwait(false);
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

    private IClientTransport CreateTransport(McpServerRegistration registration)
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
            }, _loggerFactory);
        }

        return new HttpClientTransport(new HttpClientTransportOptions
        {
            Name = registration.Alias,
            Endpoint = registration.Endpoint!,
            TransportMode = HttpTransportMode.StreamableHttp,
            ConnectionTimeout = registration.ConnectTimeout,
            AdditionalHeaders = new Dictionary<string, string>(registration.Headers),
        }, _loggerFactory);
    }

    private static async ValueTask<ConversationToolResult> ExecuteMcpAsync(
        RegisteredTool registered,
        ConversationToolCall call,
        CancellationToken cancellationToken)
    {
        McpServerRegistration server = registered.Server;
        if (Encoding.UTF8.GetByteCount(call.Arguments.GetRawText()) > server.MaxArgumentBytes)
            return ConversationToolResult.Failure(ConversationToolResultKind.ProtocolError, "arguments_too_large");

        // The call owns its arguments, so the property values stay valid for the whole request.
        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (JsonProperty property in call.Arguments.EnumerateObject())
            arguments.Add(property.Name, property.Value);

        SemaphoreSlim executionGate = registered.ExecutionGate;
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
                var taskArguments = arguments.ToDictionary(
                    static item => item.Key,
                    static item => (JsonElement)item.Value!,
                    StringComparer.Ordinal);
                dispatched = true;
                result = await registered.McpClient.CallToolWithPollingAsync(
                    new CallToolRequestParams
                    {
                        Name = registered.McpTool.ProtocolTool.Name,
                        Arguments = taskArguments,
                    },
                    server.TaskPollingIntervalMs,
                    timeout.Token).ConfigureAwait(false);
            }
            else
            {
                dispatched = true;
                result = await registered.McpTool.CallAsync(arguments, progress: null, options: null, timeout.Token).ConfigureAwait(false);
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
            return unknownOutcome
                ? ConversationToolResult.Failure(ConversationToolResultKind.UnknownOutcome, "timeout")
                : ConversationToolResult.Failure(ConversationToolResultKind.TimedOut, "timeout", isTransient: true);
        }
        finally
        {
            if (acquired)
                executionGate.Release();
        }

        if (TryDescribeUnsupportedContent(result) is { } unsupported)
        {
            return ConversationToolResult.Failure(
                ConversationToolResultKind.UnsupportedCapability,
                "unsupported_content_type",
                structuredContentJson: unsupported);
        }

        string text = string.Join("\n", result.Content.OfType<TextContentBlock>().Select(static block => block.Text));
        string? structured = result.StructuredContent is JsonElement element ? element.GetRawText() : null;
        if (Encoding.UTF8.GetByteCount(text) > server.MaxResultBytes ||
            (structured is not null && Encoding.UTF8.GetByteCount(structured) > server.MaxResultBytes))
        {
            return ConversationToolResult.Failure(ConversationToolResultKind.ProtocolError, "result_too_large");
        }

        return result.IsError == true
            ? ConversationToolResult.Failure(ConversationToolResultKind.ToolError, "mcp_tool_error", text, structured)
            : ConversationToolResult.Success(text, structured);
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

    private async Task DisposeClientsAsync()
    {
        for (int index = _clients.Count - 1; index >= 0; index--)
        {
            try
            {
                await _clients[index].DisposeAsync().ConfigureAwait(false);
            }
            catch (Exception exception)
            {
                _logger.LogWarning(exception, "Closing an MCP client connection failed.");
            }
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
        McpClientTool McpTool,
        McpClient McpClient,
        McpServerRegistration Server,
        SemaphoreSlim ExecutionGate);
}
