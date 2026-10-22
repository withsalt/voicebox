using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>
/// Adapts the current text-streaming provider at the composition edge without letting its type
/// escape into the runtime plan.
/// </summary>
internal sealed class TextOnlyResponseGeneratorFactory : IResponseGeneratorFactory
{
    private readonly IStreamLLMService _service;

    public TextOnlyResponseGeneratorFactory(IStreamLLMService service)
        => _service = service ?? throw new ArgumentNullException(nameof(service));

    public IResponseGenerator Create()
        => new LlmResponseGenerator(new TextOnlyLlmTurnClientAdapter(_service));
}
