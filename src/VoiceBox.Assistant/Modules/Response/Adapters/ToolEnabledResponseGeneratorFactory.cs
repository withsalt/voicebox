using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

internal sealed class ToolEnabledResponseGeneratorFactory(
    IToolCallingStreamLLMService service,
    IAssistantToolRuntimeFactory runtimeFactory,
    VoiceAssistantToolOptions toolOptions) : IResponseGeneratorFactory
{
    private readonly IToolCallingStreamLLMService _service = service
        ?? throw new ArgumentNullException(nameof(service));
    private readonly IAssistantToolRuntimeFactory _runtimeFactory = runtimeFactory
        ?? throw new ArgumentNullException(nameof(runtimeFactory));
    private readonly VoiceAssistantToolOptions _toolOptions = toolOptions
        ?? throw new ArgumentNullException(nameof(toolOptions));

    public IResponseGenerator Create()
        => new ToolUseOrchestrator(
            new StructuredLlmTurnClientAdapter(_service),
            _runtimeFactory.Create(
                new AssistantToolRuntimePolicy(_toolOptions.ValidateToolArguments)),
            _toolOptions);
}
