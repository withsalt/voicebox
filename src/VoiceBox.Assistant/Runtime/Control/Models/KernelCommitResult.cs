namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct KernelCommitResult(
    bool Accepted,
    bool StateChanged,
    long PublicationSequence);
