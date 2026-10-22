using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IPublicationFaultSink
{
    void Report(AssistantFailure failure);
}
