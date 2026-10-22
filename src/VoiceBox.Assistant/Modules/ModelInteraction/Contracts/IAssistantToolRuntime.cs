using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Contracts;

/// <summary>
/// Protocol-neutral tool session owned by the optional MCP integration assembly. Discovery,
/// authorization and execution use the same runtime and catalog for the session's lifetime.
/// </summary>
internal interface IAssistantToolRuntime : IAsyncDisposable
{
    Task StartAsync(CancellationToken cancellationToken);

    ValueTask<ToolCatalogSnapshot> GetSnapshotAsync(
        ToolRequestContext context,
        CancellationToken cancellationToken);

    ValueTask<ToolApprovalDecision> AuthorizeAsync(
        AssistantToolCall call,
        CancellationToken cancellationToken);

    /// <summary>Executes a call after the orchestrator has obtained an Allowed authorization.</summary>
    ValueTask<AssistantToolResult> ExecuteAsync(
        AssistantToolCall call,
        CancellationToken cancellationToken);

    Task StopAsync();
}
