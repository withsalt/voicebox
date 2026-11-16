using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Runtime.Publications.Contracts;

internal interface IAssistantPublicEventSink : IAssistantTextStreamSink
{
    long RaiseStateChanged(AssistantState from, AssistantState to);
    long RaiseWakeModeChanged(AssistantWakeMode from, AssistantWakeMode to);
    long RaiseWakeupDetected(AssistantWakeupDetectedEventArgs args);
    long RaiseUserSpoke(UserSpokeEventArgs args);
    long RaiseBargeInValidation(BargeInValidationEventArgs args);
    long RaiseFaulted(AssistantFaultedEventArgs args);
    long RaiseAssistantResponseGenerated(AssistantResponseGeneratedEventArgs args);
    long RaiseAssistantReplied(AssistantReplyEventArgs args);
}
