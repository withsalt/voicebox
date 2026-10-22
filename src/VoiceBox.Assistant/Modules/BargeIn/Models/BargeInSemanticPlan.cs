using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct BargeInSemanticPlan(
    BargeInSemanticPlanKind Kind,
    TurnSignal? Signal,
    SpeakingDecision Decision,
    BargeInTriggerSource Source,
    IReadOnlyList<float[]>? InitialAudio,
    BargeInIdentityContext Identity,
    BargeInDecisionDiagnostics Diagnostics,
    long TurnId,
    long InterruptedTurnId,
    string Reason,
    AssistantState? RestoredState,
    bool TextPresent)
{
    public static BargeInSemanticPlan InterruptionRequested(
        TurnSignal? signal,
        SpeakingDecision decision,
        BargeInTriggerSource source,
        IReadOnlyList<float[]>? initialAudio,
        BargeInIdentityContext identity,
        BargeInDecisionDiagnostics diagnostics)
        => new(
            BargeInSemanticPlanKind.InterruptionRequested,
            signal,
            decision,
            source,
            initialAudio,
            identity,
            diagnostics,
            decision.TurnId,
            decision.InterruptedTurnId,
            string.Empty,
            null,
            false);

    public static BargeInSemanticPlan AuthorizationResolved(
        TurnSignal signal,
        long turnId,
        long interruptedTurnId,
        string reason)
        => new(
            BargeInSemanticPlanKind.AuthorizationResolved,
            signal,
            default,
            default,
            null,
            default,
            default,
            turnId,
            interruptedTurnId,
            reason,
            null,
            false);

    public static BargeInSemanticPlan ValidationAccepted(
        TurnSignal signal,
        long turnId,
        string reason,
        long interruptedTurnId,
        bool textPresent)
        => new(
            BargeInSemanticPlanKind.ValidationAccepted,
            signal,
            default,
            default,
            null,
            default,
            default,
            turnId,
            interruptedTurnId,
            reason,
            null,
            textPresent);

    public static BargeInSemanticPlan ValidationRejected(
        TurnSignal signal,
        long turnId,
        string reason,
        AssistantState restoredState)
        => new(
            BargeInSemanticPlanKind.ValidationRejected,
            signal,
            default,
            default,
            null,
            default,
            default,
            turnId,
            0,
            reason,
            restoredState,
            false);

    public static BargeInSemanticPlan ValidationLateCompletion(
        TurnSignal signal,
        long turnId)
        => new(
            BargeInSemanticPlanKind.ValidationLateCompletion,
            signal,
            default,
            default,
            null,
            default,
            default,
            turnId,
            0,
            signal.Reason ?? "late-completion",
            null,
            false);
}
