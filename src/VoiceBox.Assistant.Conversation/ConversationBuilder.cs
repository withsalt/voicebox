using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 组装一个默认会话实例。它只做静态配置、校验和对象组合, 不建立网络连接。
/// </summary>
/// <remarks>
/// Build 成功后 builder 视为已使用; 拥有型工具与外部来源由返回的会话管理并释放。
/// </remarks>
public sealed class ConversationBuilder
{
    private readonly IStreamLLMService _llm;
    private readonly List<IConversationContextProvider> _contextProviders = [];
    private LlmConversationOptions _options = new();
    private bool _optionsSpecified;
    private ConversationToolsBuilder? _toolsBuilder;
    private string? _conversationId;
    private int _built;

    public ConversationBuilder(IStreamLLMService llm)
        => _llm = llm ?? throw new ArgumentNullException(nameof(llm));

    /// <summary>
    /// 指定业务会话身份。未调用时自动生成。该身份在实例生命周期内稳定。
    /// </summary>
    public ConversationBuilder UseConversationId(string conversationId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(conversationId);
        if (_conversationId is not null)
            throw new InvalidOperationException("A conversation id is already registered.");
        _conversationId = conversationId;
        return this;
    }

    public ConversationBuilder UseOptions(LlmConversationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (_optionsSpecified)
            throw new InvalidOperationException("Conversation options are already registered.");
        _options = options.Clone();
        _optionsSpecified = true;
        return this;
    }

    /// <summary>
    /// 注册一个上下文提供者。每轮在构建模型请求前按注册顺序调用一次。
    /// </summary>
    public ConversationBuilder AddContextProvider(IConversationContextProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        _contextProviders.Add(provider);
        return this;
    }

    /// <summary>
    /// 启用结构化工具。注册的工具和来源由返回的会话拥有, 并在会话释放时处置。
    /// </summary>
    public ConversationBuilder UseTools(Action<ConversationToolsBuilder> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        if (_toolsBuilder is not null)
            throw new InvalidOperationException("Tools are already configured for this builder.");

        var tools = new ConversationToolsBuilder();
        configure(tools);
        _toolsBuilder = tools;
        return this;
    }

    /// <summary>
    /// 构建会话。返回的实例由调用方负责释放; 借用的 LLM 服务不会随会话释放。
    /// </summary>
    public IConversationSession Build()
    {
        if (Interlocked.Exchange(ref _built, 1) != 0)
            throw new InvalidOperationException("This conversation builder has already been used.");

        LlmConversationOptions options = _options;
        IConversationContextProvider[] contextProviders = _contextProviders.ToArray();
        string conversationId = _conversationId ?? Guid.NewGuid().ToString("N");

        if (_toolsBuilder is null)
            return new LlmConversationSession(conversationId, options, contextProviders, _llm);

        ToolConfiguration tools = _toolsBuilder.Build();
        if (_llm is not IToolCallingStreamLLMService toolCallingService)
        {
            throw new InvalidOperationException(
                "The configured LLM service does not implement IToolCallingStreamLLMService, " +
                "so a tool-enabled conversation cannot be built.");
        }

        return new LlmToolConversationSession(
            conversationId,
            options,
            contextProviders,
            toolCallingService,
            tools);
    }
}
