using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

internal sealed record ToolDefinition
{
    public ToolDefinition(
        ConversationToolId id,
        string title,
        string description,
        OwnedJsonValue inputSchema,
        OwnedJsonValue? outputSchema,
        ConversationToolRisk risk,
        ConversationToolExecutionPolicy executionPolicy = ConversationToolExecutionPolicy.ForegroundOnly,
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

    public ConversationToolId Id { get; }

    public string Title { get; }

    public string Description { get; }

    public OwnedJsonValue InputSchema { get; }

    public OwnedJsonValue? OutputSchema { get; }

    public ConversationToolRisk Risk { get; }

    public ConversationToolExecutionPolicy ExecutionPolicy { get; }

    /// <summary>Only local tools supply a custom prompt; MCP tools keep the standard host prompt.</summary>
    public string? ExecutionPrompt { get; }

    public TimeSpan Timeout { get; }

    public static ToolDefinition From(ConversationToolDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ConversationToolDescriptor descriptor = definition.Descriptor;
        return new ToolDefinition(
            definition.Id,
            descriptor.DisplayName,
            descriptor.Description,
            OwnedJsonValue.Parse(descriptor.InputSchemaJson),
            descriptor.OutputSchemaJson is null ? null : OwnedJsonValue.Parse(descriptor.OutputSchemaJson),
            descriptor.Risk,
            descriptor.ExecutionPolicy,
            descriptor.ExecutionPrompt,
            descriptor.Timeout);
    }
}
