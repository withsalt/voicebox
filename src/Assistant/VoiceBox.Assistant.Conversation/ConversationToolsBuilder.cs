using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 为一个拥有型工具会话注册本地工具、外部工具来源和审批策略。
/// </summary>
public sealed class ConversationToolsBuilder
{
    private readonly List<IConversationTool> _tools = [];
    private readonly List<IConversationToolProvider> _providers = [];
    private IConversationToolApprovalPolicy? _approvalPolicy;
    private ConversationToolOptions _options = new();
    private bool _optionsSpecified;

    /// <summary>
    /// 注册一个进程内工具。成功 Build 后, 工具由会话拥有并在释放时处置。
    /// </summary>
    public ConversationToolsBuilder AddTool(IConversationTool tool)
    {
        ArgumentNullException.ThrowIfNull(tool);
        _tools.Add(tool);
        return this;
    }

    /// <summary>
    /// 注册一个外部工具来源 (例如 MCP)。成功 Build 后, 来源由会话拥有并在释放时处置。
    /// </summary>
    public ConversationToolsBuilder AddToolProvider(IConversationToolProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        _providers.Add(provider);
        return this;
    }

    public ConversationToolsBuilder UseApprovalPolicy(IConversationToolApprovalPolicy approvalPolicy)
    {
        ArgumentNullException.ThrowIfNull(approvalPolicy);
        if (_approvalPolicy is not null)
            throw new InvalidOperationException("A tool approval policy is already registered.");
        _approvalPolicy = approvalPolicy;
        return this;
    }

    /// <summary>
    /// Configures the default tool options and stores a validated snapshot when the callback returns.
    /// </summary>
    public ConversationToolsBuilder UseOptions(Action<ConversationToolOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        if (_optionsSpecified)
            throw new InvalidOperationException("Tool options are already registered.");

        var options = new ConversationToolOptions();
        configure(options);
        if (_optionsSpecified)
            throw new InvalidOperationException("Tool options are already registered.");

        _options = options.Clone();
        _optionsSpecified = true;
        return this;
    }

    internal ToolConfiguration Build()
    {
        if (_tools.Count == 0 && _providers.Count == 0)
            throw new InvalidOperationException("UseTools must register at least one tool or provider.");

        string? duplicateLocal = _tools
            .GroupBy(static tool => tool.Descriptor.Name, StringComparer.Ordinal)
            .FirstOrDefault(static group => group.Count() > 1)?.Key;
        if (duplicateLocal is not null)
            throw new InvalidOperationException($"Local tool '{duplicateLocal}' is already registered.");

        var aliases = new HashSet<string>(StringComparer.Ordinal) { "local" };
        foreach (IConversationToolProvider provider in _providers)
        {
            if (provider is not IConversationToolProviderMetadata metadata)
                continue;
            foreach (string alias in metadata.ServerAliases)
            {
                if (string.Equals(alias, "local", StringComparison.Ordinal))
                {
                    throw new ArgumentException("Tool provider alias 'local' is reserved for AddTool registrations.", nameof(provider));
                }
                if (!aliases.Add(alias))
                {
                    throw new InvalidOperationException($"Tool provider alias '{alias}' is already registered.");
                }
            }
        }

        return new ToolConfiguration(_tools.ToArray(), _providers.ToArray(), _approvalPolicy, _options);
    }
}

internal sealed record ToolConfiguration(
    IReadOnlyList<IConversationTool> Tools,
    IReadOnlyList<IConversationToolProvider> Providers,
    IConversationToolApprovalPolicy? ApprovalPolicy,
    ConversationToolOptions Options);
