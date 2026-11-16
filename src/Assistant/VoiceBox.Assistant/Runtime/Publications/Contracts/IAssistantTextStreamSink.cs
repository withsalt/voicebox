using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Publications.Contracts;

internal interface IAssistantTextStreamSink
{
    long RaiseAssistantText(AssistantTextEventArgs args);
    Task WaitUntilDeliveredAsync(long publicationSequence, CancellationToken cancellationToken = default);
}
