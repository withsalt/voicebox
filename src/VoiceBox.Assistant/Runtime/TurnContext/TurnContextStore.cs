using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Runtime.TurnContext;

/// <summary>
/// Runtime-owned resources for active turns. This store tracks operation/session ownership only;
/// semantic state transitions remain decisions returned by the pure TurnKernel.
/// </summary>
internal sealed class TurnContextStore : IDisposable, ITurnResourcePort
{
    private readonly object _lock = new();
    private readonly Dictionary<long, TurnContext> _contexts = [];
    private readonly TurnResourceSlot _preTurnResources = new();
    private readonly OperationCorrelationGate _correlation;
    private int _disposed;

    public TurnContextStore(OperationCorrelationGate correlation)
    {
        _correlation = correlation ?? throw new ArgumentNullException(nameof(correlation));
    }

    /// <summary>
    /// Retires every runtime-owned trace of a turn: its resources and the correlation ledger that
    /// would otherwise let a late completion reattach to it.
    /// </summary>
    public void Release(long turnId)
    {
        TryRemove(turnId);
        _correlation.ForgetTurn(turnId);
    }

    internal int Count
    {
        get
        {
            lock (_lock)
                return _contexts.Count;
        }
    }

    public bool TryCreate(long turnId)
    {
        if (turnId <= 0)
            return false;

        lock (_lock)
        {
            ThrowIfDisposed();
            return _contexts.TryAdd(turnId, new TurnContext());
        }
    }

    TurnResourceSlot ITurnResourcePort.GetOrCreate(long turnId)
    {
        if (turnId < 0)
            throw new ArgumentOutOfRangeException(nameof(turnId));

        if (turnId == 0)
            return _preTurnResources;

        lock (_lock)
        {
            ThrowIfDisposed();
            if (!_contexts.TryGetValue(turnId, out TurnContext? context))
            {
                context = new TurnContext();
                _contexts.Add(turnId, context);
            }

            return context.Resources;
        }
    }

    bool ITurnResourcePort.TryGet(long turnId, out TurnResourceSlot slot)
    {
        if (turnId < 0)
        {
            slot = null!;
            return false;
        }

        if (turnId == 0)
        {
            slot = _preTurnResources;
            return true;
        }

        lock (_lock)
        {
            ThrowIfDisposed();
            if (_contexts.TryGetValue(turnId, out TurnContext? context))
            {
                slot = context.Resources;
                return true;
            }
        }

        slot = null!;
        return false;
    }

    /// <summary>
    /// Retires a turn that never attached a response pipeline. A live response CTS stays with
    /// the store until the response owner takes it back.
    /// </summary>
    public void ReleaseIfNoResponseCts(long turnId)
    {
        CancellationTokenSource? responseCts = null;
        lock (_lock)
        {
            if (_contexts.TryGetValue(turnId, out TurnContext? context) &&
                context.Resources.ResponseCts is not null)
            {
                return;
            }

            if (_contexts.Remove(turnId, out TurnContext? removed))
            {
                responseCts = removed.Resources.ResponseCts;
                removed.Resources.ResponseCts = null;
            }
        }

        responseCts?.Dispose();
        _correlation.ForgetTurn(turnId);
    }

    bool ITurnResourcePort.TryRemoveResponseCts(CancellationTokenSource responseCts)
    {
        lock (_lock)
        {
            foreach (TurnResourceSlot slot in _contexts.Values
                .Select(static context => context.Resources)
                .Append(_preTurnResources))
            {
                if (ReferenceEquals(slot.ResponseCts, responseCts))
                {
                    slot.ResponseCts = null;
                    return true;
                }

            }

            return false;
        }
    }

    public bool TryAttachAsrSession(long turnId, long operationId, string sessionId)
    {
        ArgumentException.ThrowIfNullOrEmpty(sessionId);
        lock (_lock)
        {
            if (!_contexts.TryGetValue(turnId, out TurnContext? context) ||
                context.AsrSession is not null ||
                operationId <= 0)
            {
                return false;
            }

            context.AsrSession = new OperationResource(operationId, sessionId);
            return true;
        }
    }

    public bool TrySetRecognitionMetadata(
        long turnId,
        string? rawText,
        TimeSpan speechDuration)
    {
        lock (_lock)
        {
            if (!_contexts.TryGetValue(turnId, out TurnContext? context))
                return false;

            if (rawText is not null)
                context.RawRecognitionText = rawText;
            context.SpeechDuration = speechDuration;
            return true;
        }
    }

    public bool TryGetRecognitionMetadata(
        long turnId,
        out string? rawText,
        out TimeSpan speechDuration)
    {
        lock (_lock)
        {
            rawText = null;
            speechDuration = TimeSpan.Zero;
            if (!_contexts.TryGetValue(turnId, out TurnContext? context))
                return false;

            rawText = context.RawRecognitionText;
            speechDuration = context.SpeechDuration;
            return true;
        }
    }

    public bool TryCompleteAsrSession(long turnId, long operationId, out string? sessionId)
    {
        lock (_lock)
        {
            sessionId = null;
            if (!_contexts.TryGetValue(turnId, out TurnContext? context) ||
                context.AsrSession is not { } session ||
                session.OperationId != operationId ||
                session.Completed)
            {
                return false;
            }

            session.Completed = true;
            sessionId = session.Value;
            context.AsrSession = null;
            return true;
        }
    }

    public bool TrySetResponseCts(long turnId, CancellationTokenSource responseCts)
    {
        ArgumentNullException.ThrowIfNull(responseCts);
        lock (_lock)
        {
            if (!_contexts.TryGetValue(turnId, out TurnContext? context) || context.Resources.ResponseCts is not null)
                return false;

            context.Resources.ResponseCts = responseCts;
            return true;
        }
    }

    public bool TryTakeResponseCts(long turnId, out CancellationTokenSource? responseCts)
    {
        lock (_lock)
        {
            responseCts = null;
            if (!_contexts.TryGetValue(turnId, out TurnContext? context) || context.Resources.ResponseCts is null)
                return false;

            responseCts = context.Resources.ResponseCts;
            context.Resources.ResponseCts = null;
            return true;
        }
    }

    public bool TryRemove(long turnId)
    {
        CancellationTokenSource? responseCts = null;
        lock (_lock)
        {
            if (!_contexts.Remove(turnId, out TurnContext? context))
                return false;
            responseCts = context.Resources.ResponseCts;
            context.Resources.ResponseCts = null;
        }

        responseCts?.Dispose();
        return true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        CancellationTokenSource[] resources;
        lock (_lock)
        {
            resources = _contexts.Values
                .Select(static context => context.Resources.ResponseCts)
                .Where(static cts => cts is not null)
                .Cast<CancellationTokenSource>()
                .ToArray();
            _contexts.Clear();
        }

        foreach (CancellationTokenSource resource in resources)
            resource.Dispose();
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    private sealed class TurnContext
    {
        public OperationResource? AsrSession { get; set; }
        public TurnResourceSlot Resources { get; } = new();
        public string? RawRecognitionText { get; set; }
        public TimeSpan SpeechDuration { get; set; }
    }

    private sealed class OperationResource(long operationId, string value)
    {
        public long OperationId { get; } = operationId;
        public string Value { get; } = value;
        public bool Completed { get; set; }
    }
}
