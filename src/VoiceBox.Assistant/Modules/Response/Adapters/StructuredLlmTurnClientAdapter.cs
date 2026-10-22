using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

/// <summary>Maps the provider-neutral structured LLM contract into Assistant model-turn events.</summary>
internal sealed class StructuredLlmTurnClientAdapter(IToolCallingStreamLLMService service) : ILlmTurnClient
{
    private readonly IToolCallingStreamLLMService _service = service
        ?? throw new ArgumentNullException(nameof(service));

    public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
        ModelTurnRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        var definitions = request.ToolCatalog.Definitions.Select(definition =>
            new LlmToolDefinition(
                request.ToolCatalog.ExposedNames[definition.Id],
                definition.Description,
                ToolExecutionPreferenceSchema.AddTo(
                    definition.InputSchema.CanonicalJson,
                    definition.ExecutionPolicy),
                strict: false));
        var providerRequest = new LlmToolTurnRequest(
            request.Transcript,
            definitions,
            request.Options.ToProviderOptions());

        Dictionary<string, AssistantToolId> identities = request.ToolCatalog.ExposedNames
            .ToDictionary(static item => item.Value, static item => item.Key, StringComparer.Ordinal);

        await foreach (LlmTurnEvent turnEvent in _service.StreamTurnAsync(
            providerRequest,
            cancellationToken).ConfigureAwait(false))
        {
            switch (turnEvent)
            {
                case LlmTurnTextDelta text when text.Kind == LlmChunkKind.Thinking:
                    yield return new ThinkingDelta(text.Text);
                    break;
                case LlmTurnTextDelta text:
                    yield return new VisibleTextDelta(text.Text);
                    break;
                case LlmToolCallEvent call:
                    if (!identities.TryGetValue(call.ToolName, out AssistantToolId toolId))
                    {
                        throw new InvalidOperationException(
                            $"The model requested unknown tool '{call.ToolName}'.");
                    }

                    yield return new ToolCallRequested(
                        call.CallId,
                        toolId,
                        OwnedJsonValue.Parse(call.ArgumentsJson));
                    break;
                case LlmTurnCompletedEvent completed:
                    yield return new ModelTurnCompleted(new ModelUsage(
                        completed.PromptTokens,
                        completed.CompletionTokens,
                        completed.TotalTokens));
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unsupported provider turn event '{turnEvent.GetType().Name}'.");
            }
        }
    }
}
