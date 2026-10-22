using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Optional terminal-latch capability exposed by the production ingress. Keeping this separate
/// from the read contract lets deterministic loop fixtures remain read-only while real runtime
/// faults converge through the same ingress latch used by producer-side capacity failures.
/// </summary>
internal interface IAssistantControlIngressTerminator
{
    bool TryTerminate(AssistantFailure failure, long sourceTimestamp, out long ingressSequence);
}
