using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Conversation.Contracts;

/// <summary>
/// Removes an unanswered turn from history. This port cannot remove committed history.
/// </summary>
internal interface IConversationRollbackPort
{
    /// <summary>Removes the user message of a turn that will not be answered.</summary>
    void RemoveUnansweredUserMessage(long turnId, ChatMessage? userMessage = null);

    /// <summary>
    /// Forgets the commit-dedup record of a turn. Only the record is dropped; committed transcript
    /// entries are retained.
    /// </summary>
    void ForgetCommittedResponseTurn(long turnId);
}
