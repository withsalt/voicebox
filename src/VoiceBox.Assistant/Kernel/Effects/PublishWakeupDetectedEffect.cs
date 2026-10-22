using VoiceBox.Assistant.Kernel.Signals;

namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishWakeupDetectedEffect(
    long TurnId,
    KernelWakeupObservation Observation,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
