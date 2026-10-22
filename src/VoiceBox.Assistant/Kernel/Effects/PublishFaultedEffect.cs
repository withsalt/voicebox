using VoiceBox.Assistant.Kernel.Signals;

namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishFaultedEffect(
    long TurnId,
    KernelFailure Failure,
    long OperationId = 0)
    : PublishStatefulEventEffect(TurnId, OperationId);
