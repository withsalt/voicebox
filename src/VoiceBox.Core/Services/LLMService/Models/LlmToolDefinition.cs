namespace VoiceBox.Core.Services.LLMService.Models;

/// <summary>交给 provider 的稳定函数定义。</summary>
public sealed record LlmToolDefinition
{
    public LlmToolDefinition(
        string name,
        string description,
        string inputSchemaJson,
        bool strict = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Description = description ?? string.Empty;
        InputSchemaJson = string.IsNullOrWhiteSpace(inputSchemaJson)
            ? "{\"type\":\"object\"}"
            : inputSchemaJson;
        Strict = strict;
    }

    public string Name { get; }

    public string Description { get; }

    public string InputSchemaJson { get; }

    public bool Strict { get; }
}
