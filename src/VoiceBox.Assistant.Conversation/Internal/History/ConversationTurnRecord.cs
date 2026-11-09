using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.History;

internal sealed record ConversationTurnRecord(
        long TurnId,
        ChatMessage UserMessage,
        ChatMessage? AssistantMessage,
        IReadOnlyList<ToolCall> ToolCalls,
        IReadOnlyList<ToolResult> ToolResults,
        IReadOnlyList<ChatMessage> BackgroundAssistantMessages)
{
    /// <summary>
    /// Provider transcript slice this turn's tool rounds produced, in append order: assistant
    /// tool-call messages alternating with their correlated tool-result messages. Retained so a
    /// later turn's prompt can carry structured tool data, and dropped as a unit when the turn is
    /// trimmed or falls outside the retention window.
    /// </summary>
    public IReadOnlyList<ChatMessage> ToolMessages { get; init; } = Array.Empty<ChatMessage>();
}
