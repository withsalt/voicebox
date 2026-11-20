using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
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
    private ILoggerFactory? _loggerFactory;
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

    /// <summary>
    /// Configures the default conversation options and stores a validated snapshot when the callback returns.
    /// </summary>
    public ConversationBuilder UseOptions(Action<LlmConversationOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        if (_optionsSpecified)
            throw new InvalidOperationException("Conversation options are already registered.");

        var options = new LlmConversationOptions();
        configure(options);
        if (_optionsSpecified)
            throw new InvalidOperationException("Conversation options are already registered.");

        _options = options.Clone();
        _optionsSpecified = true;
        return this;
    }

    /// <summary>
    /// 指定会话使用的日志工厂。未调用时不输出日志。工具调用的结果与失败、后台任务故障和资源释放故障
    /// 都写入这里。日志包含工具身份、调用 ID、结果分类、耗时和异常; 会话自身不写入对话文字、工具参数或工具返回内容。
    /// </summary>
    public ConversationBuilder UseLoggerFactory(ILoggerFactory loggerFactory)
    {
        ArgumentNullException.ThrowIfNull(loggerFactory);
        if (_loggerFactory is not null)
            throw new InvalidOperationException("A logger factory is already registered.");
        _loggerFactory = loggerFactory;
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
        ILoggerFactory loggerFactory = _loggerFactory ?? NullLoggerFactory.Instance;

        if (_toolsBuilder is null)
        {
            ILogger logger = loggerFactory.CreateLogger<LlmConversationSession>();
            return new LlmConversationSession(conversationId, options, contextProviders, _llm, logger);
        }

        ToolConfiguration tools = _toolsBuilder.Build();
        if (_llm is not IToolCallingStreamLLMService toolCallingService)
        {
            throw new InvalidOperationException(
                "The configured LLM service does not implement IToolCallingStreamLLMService, " +
                "so a tool-enabled conversation cannot be built.");
        }

        ILogger toolLogger = loggerFactory.CreateLogger<LlmToolConversationSession>();
        return new LlmToolConversationSession(conversationId, options, contextProviders, toolCallingService, tools, toolLogger);
    }
}
