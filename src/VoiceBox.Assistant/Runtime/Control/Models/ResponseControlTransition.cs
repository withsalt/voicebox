namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct ResponseControlTransition(
    bool Accepted,
    bool Changed,
    long PublicationSequence);
