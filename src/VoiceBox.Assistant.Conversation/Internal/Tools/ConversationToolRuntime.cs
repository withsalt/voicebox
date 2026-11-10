using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>
/// Aggregates the local provider and every registered external source behind one catalog, one
/// approval policy, one argument validator and one execution path.
/// </summary>
internal sealed class ConversationToolRuntime : IAsyncDisposable
{
    private readonly IConversationTool[] _localTools;
    private readonly IConversationToolProvider[] _providers;
    private readonly IConversationToolApprovalPolicy? _approvalPolicy;
    private readonly ConversationToolOptions _options;
    private readonly ToolArgumentValidator _argumentValidator;
    private readonly List<IConversationToolProvider> _startedProviders = [];
    private readonly Dictionary<string, IConversationToolProvider> _providersByAlias = new(StringComparer.Ordinal);
    private ToolCatalogSnapshot? _catalog;
    private int _disposed;

    public ConversationToolRuntime(
        IReadOnlyList<IConversationTool> localTools,
        IReadOnlyList<IConversationToolProvider> providers,
        IConversationToolApprovalPolicy? approvalPolicy,
        ConversationToolOptions options)
    {
        _localTools = localTools.ToArray();
        _providers = providers.ToArray();
        _approvalPolicy = approvalPolicy;
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _argumentValidator = new ToolArgumentValidator(options.ValidateToolArguments);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_catalog is not null)
            throw new InvalidOperationException("The conversation tool runtime is already started.");

        var definitions = new List<ConversationToolDefinition>();
        try
        {
            var local = new LocalToolProvider(_localTools);
            await local.StartAsync(cancellationToken).ConfigureAwait(false);
            _providersByAlias[LocalToolProvider.LocalServerAlias] = local;
            definitions.AddRange(await local.GetToolsAsync(cancellationToken).ConfigureAwait(false));

            foreach (IConversationToolProvider provider in _providers)
            {
                await provider.StartAsync(cancellationToken).ConfigureAwait(false);
                _startedProviders.Add(provider);

                IReadOnlyList<ConversationToolDefinition> providerTools =
                    await provider.GetToolsAsync(cancellationToken).ConfigureAwait(false);
                foreach (string alias in providerTools
                    .Select(static tool => tool.Id.ServerAlias)
                    .Distinct(StringComparer.Ordinal))
                {
                    if (string.Equals(alias, LocalToolProvider.LocalServerAlias, StringComparison.Ordinal))
                    {
                        throw new InvalidOperationException($"Tool provider alias '{alias}' is reserved for AddTool registrations.");
                    }
                    if (!_providersByAlias.TryAdd(alias, provider))
                    {
                        throw new InvalidOperationException($"Tool provider alias '{alias}' is already registered.");
                    }
                }

                definitions.AddRange(providerTools);
            }

            var catalog = new ToolCatalogSnapshot(definitions.Select(ToolDefinition.From));
            if (catalog.IsEmpty)
                throw new InvalidOperationException("UseTools did not expose any tools.");
            _catalog = catalog;
        }
        catch
        {
            await StopProvidersAsync().ConfigureAwait(false);
            throw;
        }
    }

    public ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(_catalog ?? throw new InvalidOperationException("The conversation tool runtime is not started."));
    }

    public async ValueTask<ToolApprovalDecision> AuthorizeAsync(ToolCall call, CancellationToken cancellationToken)
    {
        ToolDefinition definition = FindDefinition(call.ToolId);
        if (_approvalPolicy is null)
        {
            return definition.Risk == ConversationToolRisk.ReadOnly ? ToolApprovalDecision.Allowed : ToolApprovalDecision.Denied;
        }

        ConversationToolDescriptor descriptor = ToDescriptor(definition);
        var invocation = new ConversationToolInvocation(call.CallId, call.Arguments.ToJsonElement(), call.ExecutionPreference);
        ConversationToolApproval decision = await _approvalPolicy.AuthorizeAsync(
            new ConversationToolApprovalContext(call.ToolId.ServerAlias, descriptor, invocation, call.ArgumentsHash),
            cancellationToken).ConfigureAwait(false);
        return decision == ConversationToolApproval.Allow ? ToolApprovalDecision.Allowed : ToolApprovalDecision.Denied;
    }

    public ToolResult? ValidateArguments(ToolCall call)
    {
        ToolDefinition definition = FindDefinition(call.ToolId);
        if (_argumentValidator.Validate(definition, call.Arguments) is { } violation)
        {
            return new ToolResult(
                call.CallId,
                ConversationToolResultKind.InputRequired,
                OwnedJsonValue.Parse(violation),
                "schema_validation_failed");
        }
        return null;
    }

    public async ValueTask<ToolResult> ExecuteAsync(ToolCall call, CancellationToken cancellationToken)
    {
        if (!_providersByAlias.TryGetValue(call.ToolId.ServerAlias, out IConversationToolProvider? provider))
        {
            return new ToolResult(call.CallId, ConversationToolResultKind.ProtocolError, null, "unknown_tool_provider");
        }

        ConversationToolResult result = await provider.InvokeAsync(
            new ConversationToolCall(call.CallId, call.ToolId, call.Arguments.CanonicalJson, call.ExecutionPreference),
            cancellationToken).ConfigureAwait(false);
        return ToolResult.From(result);
    }

    public async Task StopAsync()
    {
        _catalog = null;
        await StopProvidersAsync().ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        try
        {
            await StopProvidersAsync().ConfigureAwait(false);
        }
        finally
        {
            for (int index = _providers.Length - 1; index >= 0; index--)
            {
                try { await _providers[index].DisposeAsync().ConfigureAwait(false); } catch { }
            }

            // The local tools are transferred to the owning session on a successful Build and are
            // released even if startup never completed.
            var seen = new HashSet<IConversationTool>(ReferenceEqualityComparer.Instance);
            for (int index = _localTools.Length - 1; index >= 0; index--)
            {
                IConversationTool tool = _localTools[index];
                if (seen.Add(tool))
                {
                    try { tool.Dispose(); } catch { }
                }
            }
        }
    }

    private async Task StopProvidersAsync()
    {
        _catalog = null;
        _providersByAlias.Clear();
        for (int index = _startedProviders.Count - 1; index >= 0; index--)
        {
            try { await _startedProviders[index].StopAsync().ConfigureAwait(false); } catch { }
        }
        _startedProviders.Clear();
    }

    private ToolDefinition FindDefinition(ConversationToolId id)
    {
        ToolCatalogSnapshot catalog = _catalog ?? throw new InvalidOperationException("The conversation tool runtime is not started.");
        return catalog.Definitions.FirstOrDefault(definition => definition.Id == id)
            ?? throw new InvalidOperationException($"Unknown tool '{id}'.");
    }

    private static ConversationToolDescriptor ToDescriptor(ToolDefinition definition) => new(
        definition.Id.ToolName,
        definition.Title,
        definition.Description,
        definition.InputSchema.CanonicalJson,
        definition.OutputSchema?.CanonicalJson,
        definition.Risk,
        definition.ExecutionPolicy,
        ResolveExecutionPrompt(definition),
        definition.Timeout);

    private static string? ResolveExecutionPrompt(ToolDefinition definition)
        => definition.ExecutionPrompt ?? (definition.ExecutionPolicy is
            ConversationToolExecutionPolicy.BackgroundOnly or
            ConversationToolExecutionPolicy.ModelChoice
                ? CreateStandardExecutionPrompt(definition.Title)
                : null);

    private static string CreateStandardExecutionPrompt(string title)
        => $"我正在后台执行{title}任务，等会给你结果。";
}
