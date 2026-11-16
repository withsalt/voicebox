using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Kernel.Effects;

internal sealed record PublishBargeInValidationEffect(
    long TurnId,
    BargeInValidationKind ValidationKind,
    string? Text,
    string? Reason,
    long OperationId = 0,
    bool HasAcousticContext = false,
    BargeInTriggerSource Source = BargeInTriggerSource.Vad,
    BargeInDecisionDiagnostics Diagnostics = default)
    : PublishStatefulEventEffect(TurnId, OperationId);
