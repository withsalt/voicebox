namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>由宿主实现的工具审批策略。模型输出和外部来源自报的风险标注不能代替该策略。</summary>
public interface IConversationToolApprovalPolicy
{
    ValueTask<ConversationToolApproval> AuthorizeAsync(ConversationToolApprovalContext context, CancellationToken cancellationToken);
}
