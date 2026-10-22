using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

internal sealed class TextOnlyLlmTurnClientAdapter : ILlmTurnClient, IDisposable
{
    private readonly IStreamLLMService _service;
    private readonly ILLMConversation? _conversation;
    private readonly SemaphoreSlim _serializationGate = new(1, 1);
    private int _disposed;

    public TextOnlyLlmTurnClientAdapter(IStreamLLMService service)
    {
        _service = service ?? throw new ArgumentNullException(nameof(service));
        _conversation = (service as IConversationalLLMService)?.CreateConversation();
    }

    internal TextOnlyLlmTurnClientAdapter(
        IStreamLLMService service,
        ILLMConversation? conversation)
    {
        _service = service ?? throw new ArgumentNullException(nameof(service));
        _conversation = conversation;
    }

    public async IAsyncEnumerable<ModelTurnEvent> RunAsync(
        ModelTurnRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!request.ToolCatalog.IsEmpty)
        {
            throw new InvalidOperationException(
                "The current text-only model adapter does not support structured tool calls.");
        }

        await _serializationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            LlmCompleteOptions options = request.Options.ToProviderOptions();
            IAsyncEnumerable<LlmTextChunk> stream = _conversation is not null
                ? _conversation.StreamAsync(request.Transcript, options, cancellationToken)
                : _service.StreamAsync(request.Transcript, options, cancellationToken);
            bool completed = false;
            await foreach (LlmTextChunk chunk in stream
                .WithCancellation(cancellationToken)
                .ConfigureAwait(false))
            {
                yield return chunk.Kind == LlmChunkKind.Thinking
                    ? new ThinkingDelta(chunk.Text)
                    : new VisibleTextDelta(chunk.Text);

                if (chunk.IsFinal && !completed)
                {
                    completed = true;
                    yield return new ModelTurnCompleted(default);
                }
            }

            if (!completed)
                yield return new ModelTurnCompleted(default);
        }
        finally
        {
            _serializationGate.Release();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _conversation?.Dispose();
        _serializationGate.Dispose();
    }
}
