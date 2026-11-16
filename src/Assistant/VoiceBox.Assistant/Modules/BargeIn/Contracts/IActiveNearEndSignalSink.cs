using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Typed low-frequency outcomes emitted by the near-end data plane. The implementation owns
/// control ingress and terminal-fault convergence; PCM never crosses this boundary.
/// </summary>
internal interface IActiveNearEndSignalSink
{
    long NextOperationId();

    bool TryEnqueueConfirmation(long operationId);

    void EnqueueSpeechEnded(long turnId, TimeSpan duration);

    void ReportDataPlaneFault(AssistantFailure failure);
}
