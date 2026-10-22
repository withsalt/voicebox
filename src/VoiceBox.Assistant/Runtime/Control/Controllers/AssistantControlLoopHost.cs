using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>
/// Owns the lifetime of the single control-loop consumer: its stop source, its task, and the latch
/// that proves a caller is running on the consumer. It neither reads nor interprets envelopes;
/// <see cref="AssistantControlLoop"/> does the reading and dispatch.
/// </summary>
internal sealed class AssistantControlLoopHost : IDisposable
{
    private readonly AssistantControlIngress _ingress;
    private readonly TimeProvider _timeProvider;
    private readonly CancellationTokenSource _stop = new();
    private AssistantControlLoop? _loop;
    private Task _loopTask = Task.CompletedTask;
    private int _started;
    private int _mutationDepth;
    private int _terminated;

    public AssistantControlLoopHost(AssistantControlIngress ingress, TimeProvider timeProvider)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
    }

    public bool IsCompleted => _loopTask.IsCompleted;

    public bool IsTerminated => Volatile.Read(ref _terminated) != 0;

    /// <summary>True only while the calling thread is inside a consumer dispatch.</summary>
    public bool IsInMutation => Volatile.Read(ref _mutationDepth) != 0;

    /// <summary>
    /// Composition-time binding of the consumer's handler. Kept separate from construction so the
    /// handler may depend on collaborators that need the host's liveness view.
    /// </summary>
    public void Bind(IAssistantControlLoopHandler handler)
    {
        ArgumentNullException.ThrowIfNull(handler);
        if (_loop is not null)
            throw new InvalidOperationException("Assistant control loop handler was already bound.");

        _loop = new AssistantControlLoop(_ingress, handler, _timeProvider);
    }

    public void EnsureStarted()
    {
        if (Interlocked.Exchange(ref _started, 1) != 0)
            return;

        if (_loop is null)
            throw new InvalidOperationException("Assistant control loop handler was never bound.");

        _loopTask = _loop.Start(_stop.Token);
    }

    public void MarkTerminated() => Volatile.Write(ref _terminated, 1);

    /// <summary>
    /// Latches consumer-exclusive execution. A second concurrent entry means the single-consumer
    /// invariant was broken, so it throws rather than serialising silently.
    /// </summary>
    public void BeginMutation(string context)
    {
        if (Interlocked.Exchange(ref _mutationDepth, 1) != 0)
            throw new InvalidOperationException($"AssistantControlLoop received concurrent {context}.");
    }

    public void EndMutation() => Volatile.Write(ref _mutationDepth, 0);

    public void RequestStop()
    {
        try { _stop.Cancel(); }
        catch { }
    }

    public void WaitForCompletion()
    {
        try { _loopTask.GetAwaiter().GetResult(); }
        catch (OperationCanceledException) { }
    }

    public void Dispose()
    {
        try { _stop.Dispose(); }
        catch { }
    }
}
