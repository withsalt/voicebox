using VoiceBox.Assistant.Conversation.Abstractions;

namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>Session-owned channel for results produced after a request's foreground stream ended.</summary>
internal interface IBackgroundResultPublisher
{
    ValueTask PublishAsync(
        ConversationBackgroundMessage message,
        CancellationToken cancellationToken);
}
