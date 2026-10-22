using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Narrow sink that returns a correlated completion/fault signal to the single control loop.
/// Handlers never touch the ingress, the reducer, or the current snapshot.
/// </summary>
internal interface IAssistantControlSignalSink
{
    ValueTask SubmitAsync(
        AssistantSignal signal,
        AssistantSignalSource source,
        CancellationToken cancellationToken);
}
