namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct AssistantControlSignalResult(
    bool Accepted,
    bool Changed,
    long PublicationSequence);
