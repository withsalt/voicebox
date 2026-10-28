using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Buffering;

/// <summary>
/// Owns the processed-capture timeline: the bounded sample reservoir, its stream epoch, the end
/// watermark and the generation that invalidates late candidates and confirmation tails. Every
/// member is called while the near-end data plane holds its single state lock; neither this type
/// nor its reservoir takes a private lock.
/// </summary>
internal sealed class NearEndTimelineBuffer
{
    private readonly NearEndAudioReservoir _reservoir;
    private long _generation;
    private long _invalidatedGeneration;
    private long _streamEpoch;
    private long _endSample;

    public NearEndTimelineBuffer(int capacitySamples)
        => _reservoir = new NearEndAudioReservoir(capacitySamples);

    public long Generation => _generation;

    /// <summary>Generation at which the last discontinuity retired older tails and candidates.</summary>
    public long InvalidatedGeneration => _invalidatedGeneration;

    public long StreamEpoch => _streamEpoch;

    public long EndSample => _endSample;

    /// <summary>
    /// Appends one processed frame. Returns true when the timeline had to advance its generation
    /// because the frame declared a discontinuity or the source range was not contiguous.
    /// </summary>
    public bool Add(
        ReadOnlySpan<float> audio,
        in CaptureRange sourceRange,
        bool hasDiscontinuity)
    {
        bool resetExisting = _reservoir.Add(audio, in sourceRange, hasDiscontinuity);
        bool discontinuity = hasDiscontinuity || resetExisting;
        if (discontinuity)
        {
            _generation = checked(_generation + 1);
            _invalidatedGeneration = _generation;
        }

        _streamEpoch = sourceRange.StreamEpoch;
        _endSample = sourceRange.EndSampleExclusive;
        return discontinuity;
    }

    /// <summary>Clears retained audio and starts a fresh generation. Used by reset/owner stop.</summary>
    public void Clear()
    {
        _reservoir.Clear();
        _endSample = 0;
        _streamEpoch = 0;
        _generation = checked(_generation + 1);
    }

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
        => _reservoir.TryLocate(audio, out streamEpoch, out startSample, out endSample);

    public bool TryGetBounds(
        out long streamEpoch,
        out long retainedStartSample,
        out long endSample)
        => _reservoir.TryGetBounds(out streamEpoch, out retainedStartSample, out endSample);

    public bool TrySnapshotFrom(
        long streamEpoch,
        long startSample,
        int maxSamples,
        out IReadOnlyList<float[]> audio)
        => _reservoir.TrySnapshotFrom(streamEpoch, startSample, maxSamples, out audio);

    public bool TrySnapshotRange(
        long streamEpoch,
        long startSample,
        long endSampleExclusive,
        out float[] audio)
        => _reservoir.TrySnapshotRange(streamEpoch, startSample, endSampleExclusive, out audio);
}
