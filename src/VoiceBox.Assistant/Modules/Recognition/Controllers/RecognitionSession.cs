namespace VoiceBox.Assistant.Modules.Recognition.Controllers;

internal sealed class RecognitionSession
{
    private readonly object _lock = new();
    private readonly Queue<float[]> _bootstrapAudio = new();
    private readonly Queue<float[]> _pendingAudio = new();
    private string? _providerSessionId;
    private bool _finishing;
    private bool _terminal;
    private bool _audioAdmissionOpen;
    private Exception? _audioFeedFailure;
    private int _drainActive;
    private int _directFeedActive;

    public RecognitionSession(long turnId, long operationId)
    {
        if (turnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(turnId));
        if (operationId <= 0)
            throw new ArgumentOutOfRangeException(nameof(operationId));

        TurnId = turnId;
        OperationId = operationId;
    }

    public long TurnId { get; }

    public long OperationId { get; }

    public string? ProviderSessionId
    {
        get { lock (_lock) return _providerSessionId; }
    }

    public bool HasPendingAudio
    {
        get { lock (_lock) return _bootstrapAudio.Count != 0 || _pendingAudio.Count != 0; }
    }

    internal bool IsTerminal
    {
        get { lock (_lock) return _terminal; }
    }

    internal bool IsClosing
    {
        get { lock (_lock) return _finishing || _terminal; }
    }

    internal bool IsFeedActive =>
        Volatile.Read(ref _drainActive) != 0 ||
        Volatile.Read(ref _directFeedActive) != 0;

    internal Exception? AudioFeedFailure
    {
        get { lock (_lock) return _audioFeedFailure; }
    }

    internal void MarkAudioFeedFailed(Exception failure)
    {
        lock (_lock)
        {
            _audioFeedFailure ??= failure;
            _finishing = true;
            _bootstrapAudio.Clear();
            _pendingAudio.Clear();
        }
    }

    public bool TryAttach(string providerSessionId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(providerSessionId);
        lock (_lock)
        {
            if (_finishing || _terminal || _providerSessionId is not null)
                return false;

            _providerSessionId = providerSessionId;
            return true;
        }
    }

    public bool TryEnqueue(ReadOnlySpan<float> audio, int? capacity = null)
    {
        if (audio.IsEmpty)
            return true;

        lock (_lock)
        {
            if (_finishing || _terminal)
                return false;
            if (capacity is { } limit &&
                _bootstrapAudio.Count + _pendingAudio.Count >= limit)
                return false;

            _pendingAudio.Enqueue(audio.ToArray());
            return true;
        }
    }

    /// <summary>
    /// Takes ownership of an already materialized bootstrap batch. The caller must not retain or
    /// mutate the arrays after this method succeeds.
    /// </summary>
    internal bool TryEnqueueOwnedBootstrap(IReadOnlyList<float[]> audio, int capacity)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (capacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacity));
        lock (_lock)
        {
            if (_finishing || _terminal)
                return false;

            int admitted = 0;
            foreach (float[] chunk in audio)
            {
                if (chunk.Length != 0)
                    admitted++;
            }
            if (_bootstrapAudio.Count + _pendingAudio.Count + admitted > capacity)
                return false;

            foreach (float[] chunk in audio)
            {
                if (chunk.Length != 0)
                    _bootstrapAudio.Enqueue(chunk);
            }

            return true;
        }
    }

    internal void OpenAudioAdmission()
    {
        lock (_lock)
        {
            if (!_finishing && !_terminal)
                _audioAdmissionOpen = true;
        }
    }

    internal void BeginTerminalDrain()
    {
        lock (_lock)
        {
            if (_terminal)
                return;

            // Closing admission and enabling the final drain are one session-local transition.
            // Frames admitted before this lock are drained; later callbacks cannot race a provider
            // Feed after FinishSessionAsync begins.
            _finishing = true;
            _audioAdmissionOpen = true;
        }
    }

    internal bool TryBeginDirectFeed(out string sessionId)
    {
        sessionId = string.Empty;
        if (Interlocked.CompareExchange(ref _directFeedActive, 1, 0) != 0)
            return false;

        lock (_lock)
        {
            if (_finishing ||
                _terminal ||
                _providerSessionId is null ||
                !_audioAdmissionOpen ||
                _bootstrapAudio.Count != 0 ||
                _pendingAudio.Count != 0 ||
                Volatile.Read(ref _drainActive) != 0)
            {
                Volatile.Write(ref _directFeedActive, 0);
                return false;
            }

            sessionId = _providerSessionId;
            return true;
        }
    }

    internal void EndDirectFeed()
        => Volatile.Write(ref _directFeedActive, 0);

    internal bool TryBeginDrain(out string sessionId)
    {
        sessionId = string.Empty;
        if (Interlocked.CompareExchange(ref _drainActive, 1, 0) != 0)
            return false;

        lock (_lock)
        {
            if (_terminal ||
                _providerSessionId is null ||
                !_audioAdmissionOpen ||
                Volatile.Read(ref _directFeedActive) != 0)
            {
                Volatile.Write(ref _drainActive, 0);
                return false;
            }

            sessionId = _providerSessionId;
            return true;
        }
    }

    internal bool TryDequeue(out float[] audio)
    {
        lock (_lock)
        {
            if (_bootstrapAudio.Count > 0)
            {
                audio = _bootstrapAudio.Dequeue();
                return true;
            }
            if (_pendingAudio.Count > 0)
            {
                audio = _pendingAudio.Dequeue();
                return true;
            }
        }

        audio = [];
        return false;
    }

    internal void EndDrain() => Volatile.Write(ref _drainActive, 0);

    internal void MarkTerminal()
    {
        lock (_lock)
        {
            _terminal = true;
            _finishing = true;
            _bootstrapAudio.Clear();
            _pendingAudio.Clear();
        }
    }

    internal void ClearPendingAudio()
    {
        lock (_lock)
        {
            _bootstrapAudio.Clear();
            _pendingAudio.Clear();
        }
    }
}
