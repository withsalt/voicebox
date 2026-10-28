namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Narrow input port into the recognition module. ASR feed pumps, provider sessions and pending
/// PCM remain owned by Recognition; the near-end module only feeds one turn-scoped range.
/// </summary>
internal interface INearEndRecognitionAudioSink
{
    bool TryFeed(long turnId, ReadOnlySpan<float> audio);

    bool HasAttachedSession(long turnId);

    void Reset(long turnId);
}
