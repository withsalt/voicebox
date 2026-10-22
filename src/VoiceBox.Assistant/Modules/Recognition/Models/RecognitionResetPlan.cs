using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Lifecycle;

namespace VoiceBox.Assistant.Modules.Recognition.Models;

internal readonly record struct RecognitionResetPlan(
    bool Observed,
    bool IsValidationTurn,
    bool AlreadyAccepted,
    long TurnId,
    AssistantState RestoredState,
    TurnSignal? Signal)
{
    public bool HasSemanticSignal => Signal is not null;

    public static RecognitionResetPlan ValidationRejected(
        long turnId,
        AssistantState restoredState,
        TurnSignal signal)
        => new(true, true, false, turnId, restoredState, signal);

    public static RecognitionResetPlan FinalizeOnly(
        long turnId,
        bool isValidationTurn,
        bool alreadyAccepted,
        AssistantState restoredState)
        => new(true, isValidationTurn, alreadyAccepted, turnId, restoredState, null);
}
