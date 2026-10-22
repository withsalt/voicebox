using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Contracts;

/// <summary>
/// Records tool outcomes and provider exchanges for the same conversation history. Implementations
/// must be thread-safe because foreground rounds and background tool loops can overlap.
/// </summary>
internal interface IToolHistorySink
{
    /// <summary>Persists an outcome as soon as execution resolves, independently of later model rounds.</summary>
    void RecordToolActivity(
        long turnId,
        AssistantToolCall call,
        AssistantToolResult result);

    /// <summary>
    /// Receives the exact provider messages in append order. Exchanges remain pending until their
    /// turn commits; rolling back an unanswered turn removes them with its user message.
    /// </summary>
    void RecordToolExchange(long turnId, IReadOnlyList<ChatMessage> toolMessages);
}
