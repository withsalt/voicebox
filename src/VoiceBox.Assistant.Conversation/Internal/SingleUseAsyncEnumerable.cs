namespace VoiceBox.Assistant.Conversation.Internal;

/// <summary>
/// Enforces the public one-shot stream contract: a returned stream may be enumerated once. A
/// second enumeration fails instead of replaying the turn or producing a second partial result.
/// </summary>
internal sealed class SingleUseAsyncEnumerable<T>(Func<CancellationToken, IAsyncEnumerable<T>> factory) : IAsyncEnumerable<T>
{
    private readonly Func<CancellationToken, IAsyncEnumerable<T>> _factory = factory
        ?? throw new ArgumentNullException(nameof(factory));
    private int _enumerated;

    public IAsyncEnumerator<T> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        if (Interlocked.Exchange(ref _enumerated, 1) != 0)
        {
            throw new InvalidOperationException(
                "A conversation result stream can only be enumerated once.");
        }

        return _factory(cancellationToken).GetAsyncEnumerator(cancellationToken);
    }
}
