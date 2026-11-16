namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>Admits and drains turn audio using session state owned by the recognition module.</summary>
internal interface IRecognitionAudioPort
{
    bool TryFeedAudio(long turnId, ReadOnlySpan<float> audio);

    /// <summary>
    /// Reports whether capture ownership has attached a provider session for the turn.
    /// Attachment state remains owned by the recognition module; realtime callers use this only
    /// to classify a bounded-queue fault for diagnostics.
    /// </summary>
    bool HasAttachedSession(long turnId);

    /// <summary>Returns the provider session admitted for this turn before feeding its audio.</summary>
    bool TryGetProviderSession(long turnId, out long operationId, out string? sessionId);

    ValueTask DrainAudioAsync();

    Task StartAudioPump(CancellationToken cancellationToken);

    void ResetAudio(long turnId);
}
