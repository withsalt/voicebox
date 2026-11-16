using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal sealed record UserSpokePublication(UserSpokeEventArgs Args)
    : AssistantPublication;
