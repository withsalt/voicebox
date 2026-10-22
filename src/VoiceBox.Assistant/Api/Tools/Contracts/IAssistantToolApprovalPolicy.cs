using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceBox.Assistant.Api.Tools.Contracts;

/// <summary>由宿主实现的工具审批策略。模型和 MCP annotation 不能代替该策略。</summary>
public interface IAssistantToolApprovalPolicy
{
    ValueTask<AssistantToolApproval> AuthorizeAsync(AssistantToolApprovalContext context, CancellationToken cancellationToken);
}