using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

internal interface IAssistantControlLoopHandler
{
    ValueTask HandleRuntimeCommandAsync(
        RuntimeCommand command,
        ControlEnvelope envelope,
        CancellationToken cancellationToken);

    ValueTask HandleAssistantSignalAsync(
        AssistantSignal signal,
        ControlEnvelope envelope,
        CancellationToken cancellationToken);

    ValueTask HandleTerminalFaultAsync(
        AssistantFailure failure,
        ControlEnvelope envelope,
        CancellationToken cancellationToken);

    ValueTask HandleAbandonedAfterBarrierAsync(
        ControlEnvelope envelope,
        CancellationToken cancellationToken);
}
