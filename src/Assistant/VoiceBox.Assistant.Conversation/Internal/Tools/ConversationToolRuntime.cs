using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>
/// Aggregates the local tools and every registered external source behind one catalog, one
/// approval policy, one argument validator and one invocation path.
/// </summary>
internal sealed class ConversationToolRuntime : IAsyncDisposable
{
    private static readonly IReadOnlyDictionary<string, IConversationToolProvider> NoProviders =
        new Dictionary<string, IConversationToolProvider>(StringComparer.Ordinal);

    private readonly IConversationTool[] _localTools;
    private readonly IConversationToolProvider[] _providers;
    private readonly IConversationToolApprovalPolicy? _approvalPolicy;
    private readonly ToolArgumentValidator _argumentValidator;
    private readonly ILogger _logger;
    private readonly List<IConversationToolProvider> _startedProviders = [];
    // Replaced as a whole and never mutated once published: a call that outlives its turn may still read it.
    private volatile IReadOnlyDictionary<string, IConversationToolProvider> _providersByAlias = NoProviders;
    private bool _started;
    private int _disposed;

    public ConversationToolRuntime(
        IReadOnlyList<IConversationTool> localTools,
        IReadOnlyList<IConversationToolProvider> providers,
        IConversationToolApprovalPolicy? approvalPolicy,
        bool validateArguments,
        ILogger logger)
    {
        _localTools = localTools.ToArray();
        _providers = providers.ToArray();
        _approvalPolicy = approvalPolicy;
        _argumentValidator = new ToolArgumentValidator(validateArguments);
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>Starts every source and returns the catalog of this run.</summary>
    public async Task<ToolCatalogSnapshot> StartAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_started)
            throw new InvalidOperationException("The conversation tool runtime is already started.");

        var providersByAlias = new Dictionary<string, IConversationToolProvider>(StringComparer.Ordinal);
        var definitions = new List<ConversationToolDefinition>();
        try
        {
            var local = new LocalToolProvider(_localTools);
            providersByAlias[LocalToolProvider.LocalServerAlias] = local;
            definitions.AddRange(await local.GetToolsAsync(cancellationToken).ConfigureAwait(false));

            foreach (IConversationToolProvider provider in _providers)
            {
                await provider.StartAsync(cancellationToken).ConfigureAwait(false);
                _startedProviders.Add(provider);

                IReadOnlyList<ConversationToolDefinition> tools = await provider.GetToolsAsync(cancellationToken).ConfigureAwait(false);
                foreach (string alias in tools.Select(static tool => tool.Id.ServerAlias).Distinct(StringComparer.Ordinal))
                {
                    if (string.Equals(alias, LocalToolProvider.LocalServerAlias, StringComparison.Ordinal))
                        throw new InvalidOperationException($"Tool provider alias '{alias}' is reserved for AddTool registrations.");
                    if (!providersByAlias.TryAdd(alias, provider))
                        throw new InvalidOperationException($"Tool provider alias '{alias}' is already registered.");
                }

                definitions.AddRange(tools);
            }

            var catalog = new ToolCatalogSnapshot(definitions);
            if (catalog.IsEmpty)
                throw new InvalidOperationException("UseTools did not expose any tools.");

            _providersByAlias = providersByAlias;
            _started = true;
            _logger.LogDebug("Conversation tools started: {Tools} tools, {Sources} sources.", catalog.Tools.Count, providersByAlias.Count);
            return catalog;
        }
        catch
        {
            await StopProvidersAsync().ConfigureAwait(false);
            throw;
        }
    }

    /// <summary>Asks the host policy. Without a policy only read-only tools are allowed.</summary>
    public async ValueTask<bool> AuthorizeAsync(ToolCall call, ConversationToolCall request, CancellationToken cancellationToken)
    {
        if (_approvalPolicy is null)
            return call.Tool.Descriptor.Risk == ConversationToolRisk.ReadOnly;

        var context = new ConversationToolApprovalContext(call.Tool.Descriptor, request, call.ArgumentsHash);
        ConversationToolApproval decision = await _approvalPolicy.AuthorizeAsync(context, cancellationToken).ConfigureAwait(false);
        return decision == ConversationToolApproval.Allow;
    }

    /// <summary>Returns violation details for the model, or null when the arguments are acceptable.</summary>
    public string? ValidateArguments(ToolCall call) => _argumentValidator.Validate(call.Tool, call.Arguments);

    public ValueTask<ConversationToolResult> InvokeAsync(ConversationToolCall call, CancellationToken cancellationToken)
        => _providersByAlias.TryGetValue(call.ToolId.ServerAlias, out IConversationToolProvider? provider)
            ? provider.InvokeAsync(call, cancellationToken)
            : ValueTask.FromResult(ConversationToolResult.Failure(ConversationToolResultKind.ProtocolError, "unknown_tool_provider"));

    public Task StopAsync() => StopProvidersAsync();

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        await StopProvidersAsync().ConfigureAwait(false);
        for (int index = _providers.Length - 1; index >= 0; index--)
        {
            try
            {
                await _providers[index].DisposeAsync().ConfigureAwait(false);
            }
            catch (Exception exception)
            {
                _logger.LogWarning(exception, "Disposing tool provider {Provider} failed.", _providers[index].GetType().Name);
            }
        }

        // Local tools are owned by the session from a successful Build on, and are released even if it never started.
        var seen = new HashSet<IConversationTool>(ReferenceEqualityComparer.Instance);
        for (int index = _localTools.Length - 1; index >= 0; index--)
        {
            IConversationTool tool = _localTools[index];
            if (!seen.Add(tool))
                continue;
            try
            {
                tool.Dispose();
            }
            catch (Exception exception)
            {
                _logger.LogWarning(exception, "Disposing local tool {Tool} failed.", tool.GetType().Name);
            }
        }
    }

    private async Task StopProvidersAsync()
    {
        _started = false;
        _providersByAlias = NoProviders;
        for (int index = _startedProviders.Count - 1; index >= 0; index--)
        {
            try
            {
                await _startedProviders[index].StopAsync().ConfigureAwait(false);
            }
            catch (Exception exception)
            {
                _logger.LogWarning(exception, "Stopping tool provider {Provider} failed.", _startedProviders[index].GetType().Name);
            }
        }
        _startedProviders.Clear();
    }
}
