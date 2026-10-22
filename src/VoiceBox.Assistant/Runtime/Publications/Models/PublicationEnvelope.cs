namespace VoiceBox.Assistant.Runtime.Publications.Models;

internal readonly record struct PublicationEnvelope(
    long PublicationSequence,
    AssistantPublication Publication);
