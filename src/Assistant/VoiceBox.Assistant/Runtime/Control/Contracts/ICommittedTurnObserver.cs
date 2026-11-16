using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Signals;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Notified inside the commit of an accepted Kernel decision, before any externally visible effect
/// runs. Implementations may only refresh their own data-plane projection; they must not decide,
/// publish or re-enter the control loop.
/// </summary>
internal interface ICommittedTurnObserver
{
    void OnTurnCommitted(TurnSignal signal, AssistantState state);
}
