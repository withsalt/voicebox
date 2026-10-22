using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Conversation.Contracts;

/// <summary>
/// Prompt-building surface used by the response module at the start of a turn. Commit and rollback
/// of the same turn are driven by Kernel effects through the conversation effect handler, so they
/// are deliberately absent here.
/// </summary>
internal interface IConversationPromptPort
{
    /// <summary>
    /// Appends the recognized user message and returns the trimmed prompt for this turn. The
    /// returned message identifies the entry to roll back if the turn produces no answer.
    /// </summary>
    ChatMessage AddUserMessage(long turnId, string userText, out ChatMessage[] prompt);
}
