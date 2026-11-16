using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

/// <summary>
/// A model-requested call bound to its catalog entry. The arguments may still be invalid: anything that is not
/// a JSON object is reported back to the model by argument validation instead of failing the turn.
/// </summary>
internal sealed class ToolCall
{
    private string? _argumentsHash;

    public ToolCall(string callId, ToolCatalogEntry tool, JsonElement arguments, ConversationToolExecutionPreference executionPreference)
    {
        CallId = callId;
        Tool = tool;
        Arguments = arguments.Clone();
        ArgumentsJson = ModelJson.Canonicalize(Arguments);
        ExecutionPreference = executionPreference;
    }

    public string CallId { get; }

    public ToolCatalogEntry Tool { get; }

    public JsonElement Arguments { get; }

    /// <summary>Canonical argument text: written to the provider transcript and used to recognise a replayed call.</summary>
    public string ArgumentsJson { get; }

    public ConversationToolExecutionPreference ExecutionPreference { get; }

    public string ArgumentsHash => _argumentsHash ??= ModelJson.StableHash(ArgumentsJson);

    /// <summary>The public call handed to the approval policy and the tool. Requires validated object arguments.</summary>
    public ConversationToolCall ToCall() => new(CallId, Tool.Id, Arguments, ExecutionPreference);
}