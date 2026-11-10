using System.Text.Json;

namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>工具的受信任宿主元数据和模型可见 JSON Schema。</summary>
public sealed class ConversationToolDescriptor
{
    public ConversationToolDescriptor(
        string name,
        string displayName,
        string description,
        string inputSchemaJson,
        string? outputSchemaJson = null,
        ConversationToolRisk risk = ConversationToolRisk.ReadOnly,
        ConversationToolExecutionPolicy executionPolicy = ConversationToolExecutionPolicy.ModelChoice,
        string? executionPrompt = null,
        TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);

        ValidateSchema(inputSchemaJson, nameof(inputSchemaJson), requireObject: true);
        if (outputSchemaJson is not null)
            ValidateSchema(outputSchemaJson, nameof(outputSchemaJson), requireObject: false);
        if (!Enum.IsDefined(risk))
            throw new ArgumentOutOfRangeException(nameof(risk));
        if (!Enum.IsDefined(executionPolicy))
            throw new ArgumentOutOfRangeException(nameof(executionPolicy));

        TimeSpan resolvedTimeout = timeout ?? TimeSpan.FromSeconds(30);
        if (resolvedTimeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout));

        Name = name;
        DisplayName = displayName;
        Description = description ?? string.Empty;
        InputSchemaJson = inputSchemaJson;
        OutputSchemaJson = outputSchemaJson;
        Risk = risk;
        ExecutionPolicy = executionPolicy;
        ExecutionPrompt = executionPrompt;
        Timeout = resolvedTimeout;
    }

    public string Name { get; }

    /// <summary>宿主可呈现给用户的受信任名称, 不来自模型或外部来源的自报标题。</summary>
    public string DisplayName { get; }

    public string Description { get; }

    public string InputSchemaJson { get; }

    public string? OutputSchemaJson { get; }

    public ConversationToolRisk Risk { get; }

    public ConversationToolExecutionPolicy ExecutionPolicy { get; }

    /// <summary>
    /// 工具确定为后台执行时, 宿主可呈现的受信任文字。它不发送给模型, 也不由外部来源的
    /// 自报描述生成; 默认实现如何将它送入文本或语音链路由其自身文档说明。
    /// </summary>
    public string? ExecutionPrompt { get; }

    public TimeSpan Timeout { get; }

    private static void ValidateSchema(string value, string parameterName, bool requireObject)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        try
        {
            using JsonDocument document = JsonDocument.Parse(value);
            if (requireObject && document.RootElement.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("Tool input schema must be a JSON object.", parameterName);
            if (!requireObject && document.RootElement.ValueKind is not (JsonValueKind.Object or JsonValueKind.True or JsonValueKind.False))
            {
                throw new ArgumentException("Tool output schema must be a JSON object or boolean schema.", parameterName);
            }
        }
        catch (JsonException exception)
        {
            throw new ArgumentException("Tool schema is not valid JSON.", parameterName, exception);
        }
    }
}
