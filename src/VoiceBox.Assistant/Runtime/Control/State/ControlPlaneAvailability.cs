using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

namespace VoiceBox.Assistant.Runtime.Control.State;

/// <summary>
/// Single answer to "may a producer still admit control-plane work?". It combines the run scope,
/// the ingress terminal latch and the consumer's liveness so every producer applies the same
/// shutdown rule instead of re-deriving it.
/// </summary>
internal sealed class ControlPlaneAvailability
{
    private readonly RuntimeRunScope _runScope;
    private readonly AssistantControlIngress _ingress;
    private readonly AssistantControlLoopHost _loopHost;

    public ControlPlaneAvailability(
        RuntimeRunScope runScope,
        AssistantControlIngress ingress,
        AssistantControlLoopHost loopHost)
    {
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _loopHost = loopHost ?? throw new ArgumentNullException(nameof(loopHost));
    }

    public bool IsAcceptingSignals
        => !_runScope.IsDisposed &&
            !_runScope.IsCancelled &&
            !_loopHost.IsTerminated &&
            !_loopHost.IsCompleted;

    public bool IsShuttingDown => _ingress.HasTerminalFault || _runScope.IsCancelled;

    public bool CanEnqueueLifecycleCommand
        => !_ingress.HasTerminalFault && !_loopHost.IsTerminated && !_loopHost.IsCompleted;
}
