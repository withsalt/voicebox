using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal readonly record struct SpeechBoundaryObservation(
    SpeechBoundaryObservationKind Kind,
    TurnSignal? Signal,
    SpeakingDecision SpeakingDecision,
    BargeInTriggerSource BargeInSource,
    IReadOnlyList<float[]>? InitialAudio,
    TimeSpan? PendingSpeechEnd,
    long TurnId,
    string? AsrSessionId,
    TimeSpan SpeechDuration,
    BargeInIdentityContext Identity = default)
{
    public static SpeechBoundaryObservation StartListening(TurnSignal signal, long turnId)
        => new(
            SpeechBoundaryObservationKind.StartListening,
            signal,
            default,
            default,
            null,
            null,
            turnId,
            null,
            default);

    public static SpeechBoundaryObservation ConfirmedListening(
        TurnSignal signal,
        SpeakingDecision decision)
        => new(
            SpeechBoundaryObservationKind.ConfirmedListening,
            signal,
            decision,
            default,
            null,
            null,
            decision.TurnId,
            null,
            default);

    public static SpeechBoundaryObservation ConfirmedBargeIn(
        SpeakingDecision decision,
        BargeInTriggerSource source,
        IReadOnlyList<float[]>? initialAudio,
        TimeSpan? pendingSpeechEnd,
        BargeInIdentityContext identity)
        => new(
            SpeechBoundaryObservationKind.ConfirmedBargeIn,
            null,
            decision,
            source,
            initialAudio,
            pendingSpeechEnd,
            decision.TurnId,
            null,
            default,
            identity);

    public static SpeechBoundaryObservation BeginRecognition(
        long turnId,
        string sessionId,
        TimeSpan speechDuration)
        => new(
            SpeechBoundaryObservationKind.BeginRecognition,
            null,
            default,
            default,
            null,
            null,
            turnId,
            sessionId,
            speechDuration);

    public static SpeechBoundaryObservation SingleTalkAuthorization(TurnSignal signal, long turnId)
        => new(
            SpeechBoundaryObservationKind.SingleTalkAuthorization,
            signal,
            default,
            default,
            null,
            null,
            turnId,
            null,
            default);
}
