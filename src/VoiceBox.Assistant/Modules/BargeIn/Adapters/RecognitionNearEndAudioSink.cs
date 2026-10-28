using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// Adapts the recognition module's audio port to the near-end module's narrow sink. Provider
/// sessions, pending PCM and the feed pump stay owned by Recognition.
/// </summary>
internal sealed class RecognitionNearEndAudioSink : INearEndRecognitionAudioSink
{
    private readonly IRecognitionAudioPort _recognitionAudio;

    public RecognitionNearEndAudioSink(IRecognitionAudioPort recognitionAudio)
        => _recognitionAudio = recognitionAudio
            ?? throw new ArgumentNullException(nameof(recognitionAudio));

    public bool TryFeed(long turnId, ReadOnlySpan<float> audio)
        => _recognitionAudio.TryFeedAudio(turnId, audio);

    public bool HasAttachedSession(long turnId)
        => _recognitionAudio.HasAttachedSession(turnId);

    public void Reset(long turnId)
        => _recognitionAudio.ResetAudio(turnId);
}
