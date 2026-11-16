using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

internal interface IAssistantControlIngressReader
{
    bool TryRead(out ControlEnvelope envelope);

    /// <summary>
    /// Blocks the dedicated consumer thread until an envelope may be readable. The consumer
    /// owns a thread of its own, so it must not surrender it to the shared thread pool: a
    /// pool hand-off would place a real-time interruption decision behind unrelated work.
    /// </summary>
    void WaitForData(CancellationToken cancellationToken);

    ValueTask WaitForDataAsync(CancellationToken cancellationToken);
    bool IsAfterBarrier(in ControlEnvelope envelope);
}
