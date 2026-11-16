using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Control.Contracts;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// The last conversation state made visible to subscribers, across both sources that can change
/// it: Kernel decisions and the runtime's own Start/Stop transitions. TurnKernel never models
/// <see cref="AssistantState.Stopped"/>, so the lifecycle needs this mirror to decide whether a
/// start or stop actually changes anything worth publishing.
/// </summary>
internal sealed class RuntimeConversationStateMirror : ICommittedTurnObserver
{
    private readonly object _lock = new();
    private AssistantState _state = AssistantState.Idle;

    public AssistantState Current
    {
        get { lock (_lock) return _state; }
    }

    void ICommittedTurnObserver.OnTurnCommitted(TurnSignal signal, AssistantState state)
    {
        lock (_lock)
            _state = state;
    }

    /// <summary>
    /// Applies a lifecycle transition. Returns false when the state is already there, which is how
    /// a first start stays silent while a restart publishes Stopped → Idle.
    /// </summary>
    public bool TryTransition(AssistantState to, out AssistantState old)
    {
        lock (_lock)
        {
            old = _state;
            if (old == to)
                return false;

            _state = to;
            return true;
        }
    }
}
