using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Adapters;

/// <summary>
/// Standalone processed-capture timeline reservoir for compositions that verify single-talk
/// identity without an installed barge-in detector. It is a pure data-plane sink: every final
/// processed frame is retained in capture order, and the turn boundary locates VAD chunks against
/// it. When barge-in is installed the near-end coordinator's own reservoir plays this role.
/// </summary>
internal sealed class ProcessedSpeechTimelineLocator : IProcessedCaptureFrameSink, ISpeechTimelineLocator
{
    private readonly NearEndAudioReservoir _reservoir;

    public ProcessedSpeechTimelineLocator(int capacitySamples)
    {
        if (capacitySamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacitySamples));
        _reservoir = new NearEndAudioReservoir(capacitySamples);
    }

    public void OnProcessedCapture(in ProcessedCaptureFrame frame)
    {
        ReadOnlySpan<float> audio = frame.Audio.Span;
        if (audio.IsEmpty)
            return;
        CaptureRange sourceRange = frame.SourceRange;
        _reservoir.Add(audio, in sourceRange, frame.HasDiscontinuity);
    }

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
        => _reservoir.TryLocate(audio, out streamEpoch, out startSample, out endSample);
}
