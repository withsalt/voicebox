using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Runtime.Control.State;

/// <summary>
/// Lock-free copy of the committed conversation state for capture and render callbacks. Audio
/// threads read it to classify an incoming fact; they never take a decision from it. It is
/// republished at the same commit point as the snapshot, before any externally visible effect.
/// </summary>
internal sealed class TurnStateProjection : IAssistantStateProjectionSink
{
    private int _state = (int)AssistantState.Idle;

    public AssistantState Current => (AssistantState)Volatile.Read(ref _state);

    public void Publish(AssistantState state) => Volatile.Write(ref _state, (int)state);
}
