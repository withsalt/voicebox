using VoiceBox.Assistant.Modules.Wakeup.Models;

namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IWakeupDecisionSink
{
    bool TryEnqueueObservation(WakeupAudioDecision decision);
}
