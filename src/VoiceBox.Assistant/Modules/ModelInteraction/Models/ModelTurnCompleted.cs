using System.Collections.ObjectModel;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal sealed record ModelTurnCompleted : ModelTurnEvent
{
    public ModelTurnCompleted(
        ModelUsage usage,
        IEnumerable<AssistantToolCall>? toolCalls = null,
        IEnumerable<AssistantToolResult>? toolResults = null)
    {
        Usage = usage;
        ToolCalls = Array.AsReadOnly((toolCalls ?? []).ToArray());
        ToolResults = Array.AsReadOnly((toolResults ?? []).ToArray());
    }

    public ModelUsage Usage { get; }

    public ReadOnlyCollection<AssistantToolCall> ToolCalls { get; }

    public ReadOnlyCollection<AssistantToolResult> ToolResults { get; }
}
