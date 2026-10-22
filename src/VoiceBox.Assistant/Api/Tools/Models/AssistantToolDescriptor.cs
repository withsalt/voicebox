using System.Text.Json;

namespace VoiceBox.Assistant.Api.Tools.Models;

/// <summary>工具的受信任宿主元数据和模型可见 JSON Schema。</summary>
public sealed class AssistantToolDescriptor
{
    public AssistantToolDescriptor(
        string name,
        string displayName,
        string description,
        string inputSchemaJson,
        string? outputSchemaJson = null,
        AssistantToolRisk risk = AssistantToolRisk.ReadOnly,
        AssistantToolExecutionPolicy executionPolicy = AssistantToolExecutionPolicy.ModelChoice,
        string? executionPrompt = null,
        TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(displayName);
        if (name.Length > 128 || name.Any(static value =>
            !char.IsAsciiLetterOrDigit(value) && value is not '_' and not '-' and not '.'))
        {
            throw new ArgumentException(
                "Tool name must be 1-128 ASCII letters, digits, underscores, hyphens, or dots.",
                nameof(name));
        }

        ValidateSchema(inputSchemaJson, nameof(inputSchemaJson), requireObject: true);
        if (outputSchemaJson is not null)
            ValidateSchema(outputSchemaJson, nameof(outputSchemaJson), requireObject: false);
        if (!Enum.IsDefined(risk))
            throw new ArgumentOutOfRangeException(nameof(risk));
        if (!Enum.IsDefined(executionPolicy))
            throw new ArgumentOutOfRangeException(nameof(executionPolicy));
        if (executionPolicy is AssistantToolExecutionPolicy.BackgroundOnly or AssistantToolExecutionPolicy.ModelChoice)
            ArgumentException.ThrowIfNullOrWhiteSpace(executionPrompt);

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

    /// <summary>用于确认和 TTS 提示的受信任名称，不使用 MCP Server 自报 title。</summary>
    public string DisplayName { get; }

    public string Description { get; }

    public string InputSchemaJson { get; }

    public string? OutputSchemaJson { get; }

    public AssistantToolRisk Risk { get; }

    public AssistantToolExecutionPolicy ExecutionPolicy { get; }

    /// <summary>
    /// 本地工具确定为后台执行时，由宿主直接提交到 TTS/播放链路的受信任提示。
    /// 不会发送给模型，也不会从 MCP Server 元数据生成。
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
            if (!requireObject && document.RootElement.ValueKind is not
                (JsonValueKind.Object or JsonValueKind.True or JsonValueKind.False))
            {
                throw new ArgumentException(
                    "Tool output schema must be a JSON object or boolean schema.",
                    parameterName);
            }
        }
        catch (JsonException exception)
        {
            throw new ArgumentException("Tool schema is not valid JSON.", parameterName, exception);
        }
    }
}
