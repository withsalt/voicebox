namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

using VoiceBox.Assistant.Api.Tools.Models;
using InternalToolRisk = VoiceBox.Assistant.Modules.ModelInteraction.Models.AssistantToolRisk;

internal sealed record AssistantToolDefinition
{
    public AssistantToolDefinition(
        AssistantToolId id,
        string title,
        string description,
        OwnedJsonValue inputSchema,
        OwnedJsonValue? outputSchema,
        InternalToolRisk risk,
        AssistantToolExecutionPolicy executionPolicy = AssistantToolExecutionPolicy.ForegroundOnly,
        string? executionPrompt = null,
        TimeSpan? timeout = null)
    {
        if (string.IsNullOrWhiteSpace(id.ServerAlias) || string.IsNullOrWhiteSpace(id.ToolName))
            throw new ArgumentException("Tool identity must contain a server alias and tool name.", nameof(id));
        Id = id;
        Title = title ?? throw new ArgumentNullException(nameof(title));
        Description = description ?? throw new ArgumentNullException(nameof(description));
        InputSchema = inputSchema ?? throw new ArgumentNullException(nameof(inputSchema));
        OutputSchema = outputSchema;
        Risk = risk;
        ExecutionPolicy = executionPolicy;
        ExecutionPrompt = executionPrompt;
        Timeout = timeout ?? TimeSpan.FromSeconds(30);
        if (Timeout <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(timeout));
    }

    public AssistantToolId Id { get; }

    public string Title { get; }

    public string Description { get; }

    public OwnedJsonValue InputSchema { get; }

    public OwnedJsonValue? OutputSchema { get; }

    public InternalToolRisk Risk { get; }

    public AssistantToolExecutionPolicy ExecutionPolicy { get; }

    /// <summary>
    /// 仅本地 IAssistantTool 提供自定义值；标准 MCP 工具保持为 null 并使用既有 Host 提示。
    /// </summary>
    public string? ExecutionPrompt { get; }

    public TimeSpan Timeout { get; }
}
