namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>One executor event before the session finalizes and commits the turn.</summary>
internal abstract record TurnEvent;

internal sealed record TurnThinkingDelta(string Text) : TurnEvent;

internal sealed record TurnTextDelta(string Text, bool EndsTextSegment = false) : TurnEvent;

/// <summary>
/// The executor has produced its final answer. The session commits exactly once, releases the
/// optional background acceptance gate, then publishes the public completed update.
/// </summary>
internal sealed record TurnFinished(TurnCommitGate? CommitGate = null) : TurnEvent;

/// <summary>Per-turn input passed from the session to its executor.</summary>
internal sealed record TurnExecutionContext(
    long TurnId,
    Guid RequestId,
    string ConversationId,
    IReadOnlyList<VoiceBox.Core.Services.LLMService.Models.ChatMessage> Prompt,
    ModelGenerationOptions Options);
