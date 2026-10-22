using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using ModelContextProtocol;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Extensions.Tasks;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using InternalToolRisk = VoiceBox.Assistant.Modules.ModelInteraction.Models.AssistantToolRisk;
using PublicToolRisk = VoiceBox.Assistant.Api.Tools.Models.AssistantToolRisk;
using VoiceBox.Assistant.Api.Tools.Models;
using VoiceBox.Assistant.Api.Tools.Contracts;

namespace VoiceBox.Assistant.Integrations.Mcp;

internal sealed class McpToolRuntime : IAssistantToolRuntime
{
    private const string LocalServerAlias = "local";

    private readonly IAssistantTool[] _localTools;
    private readonly McpServerRegistration[] _servers;
    private readonly IAssistantToolApprovalPolicy? _approvalPolicy;
    private readonly ToolArgumentValidator? _argumentValidator;
    private readonly Dictionary<AssistantToolId, RegisteredTool> _tools = [];
    private readonly List<McpClient> _clients = [];
    private readonly List<SemaphoreSlim> _serverGates = [];
    private ToolCatalogSnapshot? _catalog;
    private int _disposed;

    public McpToolRuntime(
        IReadOnlyList<IAssistantTool> localTools,
        IReadOnlyList<McpServerRegistration> servers,
        IAssistantToolApprovalPolicy? approvalPolicy,
        bool validateArguments = true)
    {
        _localTools = localTools.ToArray();
        _servers = servers.ToArray();
        _approvalPolicy = approvalPolicy;
        _argumentValidator = validateArguments ? new ToolArgumentValidator() : null;
        if (_localTools.Length > 0 &&
            _servers.Any(static server => server.Alias == LocalServerAlias))
        {
            throw new InvalidOperationException(
                $"MCP server alias '{LocalServerAlias}' is reserved for AddTool registrations.");
        }
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_catalog is not null)
            throw new InvalidOperationException("MCP tool runtime is already started.");

        _tools.Clear();
        try
        {
            AddLocalTools();
            foreach (McpServerRegistration registration in _servers)
                await AddServerAsync(registration, cancellationToken).ConfigureAwait(false);

            _catalog = new ToolCatalogSnapshot(_tools.Values.Select(static value => value.Definition));
        }
        catch
        {
            await DisposeClientsAsync().ConfigureAwait(false);
            DisposeServerGates();
            _tools.Clear();
            throw;
        }
    }

    public async Task StopAsync()
    {
        _catalog = null;
        _tools.Clear();
        await DisposeClientsAsync().ConfigureAwait(false);
        DisposeServerGates();
    }

    public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
        ToolRequestContext context,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(_catalog
            ?? throw new InvalidOperationException("MCP tool runtime is not started."));
    }

    public async ValueTask<ToolApprovalDecision> AuthorizeAsync(
        AssistantToolCall call,
        CancellationToken cancellationToken)
    {
        RegisteredTool tool = GetTool(call.ToolId);
        if (_approvalPolicy is null)
        {
            return tool.Definition.Risk == InternalToolRisk.ReadOnly
                ? ToolApprovalDecision.Allowed
                : ToolApprovalDecision.Denied;
        }

        AssistantToolDescriptor descriptor = ToPublicDescriptor(tool.Definition);
        var invocation = new AssistantToolInvocation(
            call.CallId,
            call.Arguments.ToJsonElement(),
            call.ExecutionPreference);
        AssistantToolApproval decision = await _approvalPolicy.AuthorizeAsync(
            new AssistantToolApprovalContext(
                call.ToolId.ServerAlias,
                descriptor,
                invocation,
                call.ArgumentsHash),
            cancellationToken).ConfigureAwait(false);
        return decision == AssistantToolApproval.Allow
            ? ToolApprovalDecision.Allowed
            : ToolApprovalDecision.Denied;
    }

    public async ValueTask<AssistantToolResult> ExecuteAsync(
        AssistantToolCall call,
        CancellationToken cancellationToken)
    {
        RegisteredTool registered = GetTool(call.ToolId);
        // Schema validation happens before any tool code runs, so an argument mistake never reaches
        // a tool implementation and comes back to the model as actionable detail instead of an
        // opaque exception name.
        if (_argumentValidator?.Validate(registered.Definition, call.Arguments) is { } violation)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.InputRequired,
                OwnedJsonValue.Parse(violation),
                "schema_validation_failed");
        }

        if (registered.LocalTool is not null)
            return await ExecuteLocalAsync(registered.LocalTool, call, cancellationToken).ConfigureAwait(false);
        return await ExecuteMcpAsync(registered, call, cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            await StopAsync().ConfigureAwait(false);
        }
        finally
        {
            // The assistant drains foreground and background calls before disposing this runtime.
            // Release tools even if startup never completed or another cleanup step failed.
            var seen = new HashSet<IAssistantTool>(ReferenceEqualityComparer.Instance);
            for (int index = _localTools.Length - 1; index >= 0; index--)
            {
                IAssistantTool tool = _localTools[index];
                if (seen.Add(tool))
                {
                    try { tool.Dispose(); } catch { }
                }
            }
        }
    }

    private void AddLocalTools()
    {
        foreach (IAssistantTool tool in _localTools)
        {
            AssistantToolDescriptor descriptor = tool.Descriptor
                ?? throw new InvalidOperationException("A local tool returned a null descriptor.");
            var id = new AssistantToolId(LocalServerAlias, descriptor.Name);
            AddTool(new RegisteredTool(
                ToInternalDefinition(id, descriptor),
                tool,
                McpTool: null,
                McpClient: null,
                Server: null,
                ExecutionGate: null));
        }
    }

    private async Task AddServerAsync(
        McpServerRegistration registration,
        CancellationToken cancellationToken)
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
        var executionGate = new SemaphoreSlim(
            registration.MaxConcurrentCalls,
            registration.MaxConcurrentCalls);
        _serverGates.Add(executionGate);
        using var discovery = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        discovery.CancelAfter(registration.RequestTimeout);
        IList<McpClientTool> discovered = await client.ListToolsAsync(
            options: null,
            discovery.Token).ConfigureAwait(false);
        foreach (McpClientTool mcpTool in discovered)
        {
            if (registration.AllowedTools.Count > 0 &&
                !registration.AllowedTools.Contains(mcpTool.Name))
            {
                continue;
            }

            Tool protocol = mcpTool.ProtocolTool;
            string inputSchema = protocol.InputSchema.GetRawText();
            string? outputSchema = protocol.OutputSchema?.GetRawText();
            var descriptor = new AssistantToolDefinition(
                new AssistantToolId(registration.Alias, protocol.Name),
                registration.DisplayName,
                protocol.Description ?? string.Empty,
                OwnedJsonValue.Parse(inputSchema),
                outputSchema is null ? null : OwnedJsonValue.Parse(outputSchema),
                ToInternalRisk(registration.DefaultRisk),
                registration.ExecutionPolicy,
                executionPrompt: null,
                timeout: registration.RequestTimeout);
            AddTool(new RegisteredTool(
                descriptor,
                null,
                mcpTool,
                client,
                registration,
                executionGate));
        }
    }

    private static IClientTransport CreateTransport(McpServerRegistration registration)
    {
        if (registration.Transport == McpServerTransport.Stdio)
        {
            Dictionary<string, string?> environment =
                StdioClientTransportOptions.GetDefaultEnvironmentVariables();
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

    private async ValueTask<AssistantToolResult> ExecuteLocalAsync(
        IAssistantTool tool,
        AssistantToolCall call,
        CancellationToken cancellationToken)
    {
        AssistantToolExecutionResult result = await tool.ExecuteAsync(
            new AssistantToolInvocation(
                call.CallId,
                call.Arguments.ToJsonElement(),
                call.ExecutionPreference),
            cancellationToken).ConfigureAwait(false);
        OwnedJsonValue value = NormalizeLocalResult(result);
        return new AssistantToolResult(
            call.CallId,
            result.IsSuccess ? AssistantToolResultKind.Success : AssistantToolResultKind.ToolError,
            value,
            result.ErrorCode);
    }

    private static OwnedJsonValue NormalizeLocalResult(AssistantToolExecutionResult result)
    {
        if (!string.IsNullOrWhiteSpace(result.StructuredContentJson))
            return OwnedJsonValue.Parse(result.StructuredContentJson);
        return OwnedJsonValue.Parse(JsonSerializer.Serialize(new { content = result.Content }));
    }

    private static async ValueTask<AssistantToolResult> ExecuteMcpAsync(
        RegisteredTool registered,
        AssistantToolCall call,
        CancellationToken cancellationToken)
    {
        McpClientTool tool = registered.McpTool
            ?? throw new InvalidOperationException("MCP tool registration has no client tool.");
        McpServerRegistration server = registered.Server
            ?? throw new InvalidOperationException("MCP tool registration has no server options.");
        if (Encoding.UTF8.GetByteCount(call.Arguments.CanonicalJson) > server.MaxArgumentBytes)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.ProtocolError,
                null,
                "arguments_too_large");
        }

        var arguments = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (JsonProperty property in call.Arguments.ToJsonElement().EnumerateObject())
            arguments.Add(property.Name, property.Value.Clone());

        SemaphoreSlim executionGate = registered.ExecutionGate
            ?? throw new InvalidOperationException("MCP tool registration has no execution gate.");
        await executionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        CallToolResult result;
        try
        {
            if (server.UseMcpTasks)
            {
                McpClient client = registered.McpClient
                    ?? throw new InvalidOperationException("MCP task tool registration has no client.");
                var taskArguments = arguments.ToDictionary(
                    static item => item.Key,
                    static item => (JsonElement)item.Value!,
                    StringComparer.Ordinal);
                result = await client.CallToolWithPollingAsync(
                    new CallToolRequestParams
                    {
                        Name = tool.ProtocolTool.Name,
                        Arguments = taskArguments,
                    },
                    server.TaskPollingIntervalMs,
                    cancellationToken).ConfigureAwait(false);
            }
            else
            {
                result = await tool.CallAsync(
                    arguments,
                    progress: null,
                    options: null,
                    cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            executionGate.Release();
        }
        if (TryDescribeUnsupportedContent(result) is { } unsupported)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.UnsupportedCapability,
                OwnedJsonValue.Parse(unsupported),
                "unsupported_content_type");
        }

        string resultJson = NormalizeMcpResult(result);
        if (Encoding.UTF8.GetByteCount(resultJson) > server.MaxResultBytes)
        {
            return new AssistantToolResult(
                call.CallId,
                AssistantToolResultKind.ProtocolError,
                null,
                "result_too_large");
        }

        return new AssistantToolResult(
            call.CallId,
            result.IsError == true ? AssistantToolResultKind.ToolError : AssistantToolResultKind.Success,
            OwnedJsonValue.Parse(resultJson),
            result.IsError == true ? "mcp_tool_error" : null);
    }

    /// <summary>
    /// Reports a result this release cannot represent for the model. Returning an explicit
    /// unsupported-capability outcome keeps the promise that such a result is neither silently
    /// awaited, auto-fetched, nor reported as success. Structured content and text are supported, so
    /// only a result whose payload is exclusively non-text blocks is rejected.
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

        string[] text = result.Content.OfType<TextContentBlock>()
            .Select(static block => block.Text)
            .ToArray();
        return JsonSerializer.Serialize(new { content = string.Join("\n", text) });
    }

    private void AddTool(RegisteredTool tool)
    {
        if (!_tools.TryAdd(tool.Definition.Id, tool))
        {
            throw new InvalidOperationException(
                $"Duplicate tool identity '{tool.Definition.Id.ServerAlias}/{tool.Definition.Id.ToolName}'.");
        }
    }

    private RegisteredTool GetTool(AssistantToolId id)
        => _tools.TryGetValue(id, out RegisteredTool? tool)
            ? tool
            : throw new InvalidOperationException($"Unknown tool '{id.ServerAlias}/{id.ToolName}'.");

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

    private static AssistantToolDefinition ToInternalDefinition(
        AssistantToolId id,
        AssistantToolDescriptor descriptor) => new(
            id,
            descriptor.DisplayName,
            descriptor.Description,
            OwnedJsonValue.Parse(descriptor.InputSchemaJson),
            descriptor.OutputSchemaJson is null
                ? null
                : OwnedJsonValue.Parse(descriptor.OutputSchemaJson),
            ToInternalRisk(descriptor.Risk),
            descriptor.ExecutionPolicy,
            descriptor.ExecutionPrompt,
            descriptor.Timeout);

    private static AssistantToolDescriptor ToPublicDescriptor(AssistantToolDefinition definition) => new(
        definition.Id.ToolName,
        definition.Title,
        definition.Description,
        definition.InputSchema.CanonicalJson,
        definition.OutputSchema?.CanonicalJson,
        ToPublicRisk(definition.Risk),
        definition.ExecutionPolicy,
        ResolvePublicExecutionPrompt(definition),
        definition.Timeout);

    private static string? ResolvePublicExecutionPrompt(AssistantToolDefinition definition)
        => definition.ExecutionPrompt ?? (definition.ExecutionPolicy is
            AssistantToolExecutionPolicy.BackgroundOnly or
            AssistantToolExecutionPolicy.ModelChoice
                ? CreateStandardExecutionPrompt(definition.Title)
                : null);

    private static string CreateStandardExecutionPrompt(string title)
        => $"我正在后台执行{title}任务，等会给你结果。";

    private static InternalToolRisk ToInternalRisk(PublicToolRisk risk) => risk switch
    {
        PublicToolRisk.ReadOnly => InternalToolRisk.ReadOnly,
        PublicToolRisk.Sensitive => InternalToolRisk.Sensitive,
        PublicToolRisk.SideEffecting => InternalToolRisk.SideEffecting,
        PublicToolRisk.Destructive => InternalToolRisk.Destructive,
        PublicToolRisk.Privileged => InternalToolRisk.Privileged,
        _ => throw new ArgumentOutOfRangeException(nameof(risk), risk, null),
    };

    private static PublicToolRisk ToPublicRisk(InternalToolRisk risk) => risk switch
    {
        InternalToolRisk.ReadOnly => PublicToolRisk.ReadOnly,
        InternalToolRisk.Sensitive => PublicToolRisk.Sensitive,
        InternalToolRisk.SideEffecting => PublicToolRisk.SideEffecting,
        InternalToolRisk.Destructive => PublicToolRisk.Destructive,
        InternalToolRisk.Privileged => PublicToolRisk.Privileged,
        _ => throw new ArgumentOutOfRangeException(nameof(risk), risk, null),
    };

    private sealed record RegisteredTool(
        AssistantToolDefinition Definition,
        IAssistantTool? LocalTool,
        McpClientTool? McpTool,
        McpClient? McpClient,
        McpServerRegistration? Server,
        SemaphoreSlim? ExecutionGate);
}
