using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>Maps the provider-neutral structured LLM contract into executor turn events.</summary>
internal sealed class StructuredLlmTurnClient(IToolCallingStreamLLMService service)
{
    private readonly IToolCallingStreamLLMService _service = service ?? throw new ArgumentNullException(nameof(service));

    public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
        ModelTurnContext request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        var definitions = request.Catalog.Definitions.Select(definition =>
            new LlmToolDefinition(
                request.Catalog.ExposedNames[definition.Id],
                definition.Description,
                ToolExecutionPreferenceSchema.AddTo(definition.InputSchema.CanonicalJson, definition.ExecutionPolicy),
                strict: false));
        var providerRequest = new LlmToolTurnRequest(request.Transcript, definitions, request.Options.ToProviderOptions());

        Dictionary<string, ConversationToolId> identities =
            request.Catalog.ExposedNames.ToDictionary(static item => item.Value, static item => item.Key, StringComparer.Ordinal);

        await foreach (LlmTurnEvent turnEvent in _service.StreamTurnAsync(providerRequest, cancellationToken).ConfigureAwait(false))
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
                    if (!identities.TryGetValue(call.ToolName, out var toolId))
                    {
                        throw new InvalidOperationException($"The model requested unknown tool '{call.ToolName}'.");
                    }

                    yield return new ModelTurnToolCall(call.CallId, toolId, OwnedJsonValue.Parse(call.ArgumentsJson));
                    break;
                case LlmTurnCompletedEvent:
                    yield return new ModelTurnCompleted();
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported provider turn event '{turnEvent.GetType().Name}'.");
            }
        }
    }
}

internal sealed class ModelTurnContext(
    IReadOnlyList<ChatMessage> transcript,
    ModelGenerationOptions options,
    ToolCatalogSnapshot catalog,
    long turnId,
    Guid requestId)
{
    public IReadOnlyList<ChatMessage> Transcript { get; } = transcript ?? throw new ArgumentNullException(nameof(transcript));

    public IReadOnlyList<ChatMessage> TranscriptSnapshot => Transcript;

    public ModelGenerationOptions Options { get; } = options;

    public ToolCatalogSnapshot Catalog { get; } = catalog ?? throw new ArgumentNullException(nameof(catalog));

    public long TurnId { get; } = turnId;

    public Guid RequestId { get; } = requestId;
}
