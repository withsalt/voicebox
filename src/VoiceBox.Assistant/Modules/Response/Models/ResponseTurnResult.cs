using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Response.Models;

internal readonly record struct ResponseTurnResult(
    ChatMessage? UserMessage,
    bool EndConversationRequested);
