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

internal readonly record struct RecognitionSemanticPlan(
    RecognitionSemanticPlanKind Kind,
    TurnSignal? Signal,
    long TurnId,
    string SessionId,
    TimeSpan SpeechDuration)
{
    public static RecognitionSemanticPlan SessionReady(
        long turnId,
        string sessionId,
        TimeSpan speechDuration,
        TurnSignal signal)
        => new(RecognitionSemanticPlanKind.SessionReady, signal, turnId, sessionId, speechDuration);

    public static RecognitionSemanticPlan SpeechEnded(
        long turnId,
        string sessionId,
        TimeSpan speechDuration,
        TurnSignal signal)
        => new(RecognitionSemanticPlanKind.SpeechEnded, signal, turnId, sessionId, speechDuration);
}
