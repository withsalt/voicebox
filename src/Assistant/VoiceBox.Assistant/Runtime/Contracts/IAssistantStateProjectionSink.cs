using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IAssistantStateProjectionSink
{
    void Publish(AssistantState state);
}
