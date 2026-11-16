using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 一轮请求开始前的不可变上下文快照。
/// </summary>
/// <param name="ConversationId">业务会话身份。</param>
/// <param name="Request">本轮请求。</param>
/// <param name="CommittedHistory">已提交的模型历史快照 (不含 system 消息)。每次调用都是复制数据。</param>
public sealed record ConversationContext(string ConversationId, ConversationRequest Request, IReadOnlyList<ChatMessage> CommittedHistory);
