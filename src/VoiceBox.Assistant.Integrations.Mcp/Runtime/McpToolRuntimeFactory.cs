using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Integrations.Mcp;

internal sealed class McpToolRuntimeFactory(
    IReadOnlyList<IAssistantTool> localTools,
    IReadOnlyList<McpServerRegistration> servers,
    IAssistantToolApprovalPolicy? approvalPolicy) : IAssistantToolRuntimeFactory
{
    private readonly IAssistantTool[] _localTools = localTools.ToArray();
    private readonly McpServerRegistration[] _servers = servers.ToArray();

    public IAssistantToolRuntime Create(AssistantToolRuntimePolicy policy)
        => new McpToolRuntime(_localTools, _servers, approvalPolicy, policy.ValidateArguments);
}
