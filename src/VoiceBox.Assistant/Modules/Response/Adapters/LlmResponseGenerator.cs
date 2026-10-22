using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.ModelInteraction.Models;
using VoiceBox.Assistant.Modules.Response.Contracts;

namespace VoiceBox.Assistant.Modules.Response.Adapters;

internal sealed class LlmResponseGenerator : IResponseGenerator, IDisposable
{
    private readonly ILlmTurnClient _client;
    private int _disposed;

    public LlmResponseGenerator(ILlmTurnClient client)
        => _client = client ?? throw new ArgumentNullException(nameof(client));

    public IAsyncEnumerable<ModelTurnEvent> GenerateAsync(
        ModelTurnRequest request,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _client.RunAsync(request, cancellationToken);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_client is IDisposable disposable)
            disposable.Dispose();
    }
}
