using System.Runtime.CompilerServices;
using System.Text.Json;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Internal.Tools;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>Runs one model round and maps the provider-neutral structured LLM contract into executor events.</summary>
internal sealed class StructuredLlmTurnClient(IToolCallingStreamLLMService service)
{
    private readonly IToolCallingStreamLLMService _service = service ?? throw new ArgumentNullException(nameof(service));

    public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
        IReadOnlyList<ChatMessage> transcript,
        ToolCatalogSnapshot catalog,
        ModelGenerationOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var request = new LlmToolTurnRequest(transcript, catalog.ModelDefinitions, options.ToProviderOptions());
        await foreach (LlmTurnEvent turnEvent in _service.StreamTurnAsync(request, cancellationToken).ConfigureAwait(false))
        {
            switch (turnEvent)
            {
                case LlmTurnTextDelta { Kind: LlmChunkKind.Thinking } thinking:
                    yield return new ThinkingDelta(thinking.Text);
                    break;
                case LlmTurnTextDelta text:
                    yield return new VisibleTextDelta(text.Text);
                    break;
                case LlmToolCallEvent call:
                    yield return new ModelTurnToolCall(Resolve(call, catalog));
                    break;
                case LlmTurnCompletedEvent:
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported provider turn event '{turnEvent.GetType().Name}'.");
            }
        }
    }

    private static ToolCall Resolve(LlmToolCallEvent call, ToolCatalogSnapshot catalog)
    {
        if (string.IsNullOrWhiteSpace(call.CallId))
            throw new InvalidOperationException("The provider returned a tool call without a call id.");
        if (!catalog.TryGetByExposedName(call.ToolName, out ToolCatalogEntry? tool))
            throw new InvalidOperationException($"The model requested unknown tool '{call.ToolName}'.");

        using JsonDocument document = JsonDocument.Parse(string.IsNullOrWhiteSpace(call.ArgumentsJson) ? "{}" : call.ArgumentsJson);
        (JsonElement arguments, ConversationToolExecutionPreference preference) =
            ToolExecutionPreferenceSchema.Extract(document.RootElement, tool.Descriptor.ExecutionPolicy);
        return new ToolCall(call.CallId, tool, arguments, preference);
    }
}
