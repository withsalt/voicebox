using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Diagnostics;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Buffering;

/// <summary>
/// Pre-allocated identity-audio ring keyed by interleaved source sample. <see cref="Append"/> only
/// validates, copies and updates bounds under a short lock; it never calls a provider. A gap starts
/// a new contiguous segment at the break: audio after the gap stays fully queryable while any
/// request that would span the gap fails with <see cref="SpeakerTimelineFailure.TimelineGap"/>.
/// </summary>
internal sealed class SpeakerAudioTimelineBuffer
{
    private readonly float[] _ring;
    private readonly int _sampleRate;
    private readonly SpeakerVerificationDiagnostics? _diagnostics;
    private readonly object _lock = new();
    private long _streamEpoch;
    private long _retainedStart;
    private long _segmentStart;
    private long _end;
    private int _gapGeneration;
    private bool _hasStream;

    public SpeakerAudioTimelineBuffer(
        int capacitySamples,
        int sampleRate,
        SpeakerVerificationDiagnostics? diagnostics = null)
    {
        if (capacitySamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacitySamples));
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        _ring = new float[capacitySamples];
        _sampleRate = sampleRate;
        _diagnostics = diagnostics;
    }

    public int Capacity => _ring.Length;

    public int SampleRate => _sampleRate;

    /// <summary>Incremented on every discontinuity; lets a consumer notice a gap without re-reading bounds.</summary>
    public int GapGeneration => Volatile.Read(ref _gapGeneration);

    /// <summary>
    /// Copies one identity frame. Returns false when the frame does not match the ring format or
    /// carries no valid source range; such frames are also recorded as a gap.
    /// </summary>
    public bool Append(in SpeakerCaptureFrame frame)
    {
        if (!frame.IsValid || frame.SampleRate != _sampleRate || frame.Channels != 1)
        {
            lock (_lock)
                MarkGapLocked();
            _diagnostics?.RecordInvalidFrame();
            return false;
        }

        ReadOnlySpan<float> audio = frame.Audio.Span;
        CaptureRange range = frame.SourceRange;
        lock (_lock)
        {
            bool newStream = !_hasStream || range.StreamEpoch != _streamEpoch;
            if (newStream)
            {
                _hasStream = true;
                _streamEpoch = range.StreamEpoch;
                _retainedStart = range.StartSample;
                _segmentStart = range.StartSample;
                _end = range.StartSample;
                _gapGeneration++;
            }
            else if (frame.HasDiscontinuity || range.StartSample != _end)
            {
                long missing = Math.Max(0, range.StartSample - _end);
                // A hole (or a rewind) in the same epoch: retained audio before the break stays
                // valid for its own segment, but no window may cross the break.
                if (range.StartSample < _end)
                {
                    _retainedStart = range.StartSample;
                    _end = range.StartSample;
                }
                _segmentStart = range.StartSample;
                _gapGeneration++;
                _diagnostics?.RecordTimelineGap(missing);
            }

            int length = audio.Length;
            if (length >= _ring.Length)
            {
                // Only the newest ring-sized suffix can be retained.
                audio = audio.Slice(length - _ring.Length);
                long newStart = range.EndSampleExclusive - _ring.Length;
                _retainedStart = newStart;
                CopyIn(audio, newStart);
            }
            else
            {
                CopyIn(audio, range.StartSample);
            }

            _end = range.EndSampleExclusive;
            long retentionFloor = _end - _ring.Length;
            if (_retainedStart < retentionFloor)
                _retainedStart = retentionFloor;
            // Retention is not a discontinuity. Keep the hop grid anchored at the real segment
            // origin, even when the oldest PCM is overwritten by a non-hop-sized callback.
            _diagnostics?.RecordAppendedFrame();
            return true;
        }
    }

    public bool TryGetBounds(out long streamEpoch, out long segmentStart, out long retainedStart, out long end)
    {
        lock (_lock)
        {
            streamEpoch = _streamEpoch;
            segmentStart = _segmentStart;
            retainedStart = _retainedStart;
            end = _end;
            return _hasStream;
        }
    }

    /// <summary>
    /// Copies <c>[start, end)</c> of <paramref name="streamEpoch"/> into <paramref name="destination"/>.
    /// Succeeds only when the interval lies entirely inside the current contiguous segment.
    /// </summary>
    public bool TrySnapshot(
        long streamEpoch,
        long start,
        long end,
        Span<float> destination,
        out SpeakerTimelineFailure failure)
    {
        if (end <= start || destination.Length < end - start)
        {
            failure = SpeakerTimelineFailure.RangeInvalid;
            return false;
        }

        lock (_lock)
        {
            failure = ClassifyLocked(streamEpoch, start, end);
            if (failure != SpeakerTimelineFailure.None)
                return false;

            CopyOut(destination.Slice(0, checked((int)(end - start))), start);
            return true;
        }
    }

    /// <summary>Classifies an interval without copying.</summary>
    public SpeakerTimelineFailure Classify(long streamEpoch, long start, long end)
    {
        if (end <= start)
            return SpeakerTimelineFailure.RangeInvalid;
        lock (_lock)
            return ClassifyLocked(streamEpoch, start, end);
    }

    /// <summary>
    /// Counts samples in <c>[start, end)</c> belonging to 10ms frames whose RMS reaches
    /// <paramref name="minFrameRms"/>. Used for the speech-time gate; never allocates.
    /// </summary>
    public bool TryCountSpeechSamples(
        long streamEpoch,
        long start,
        long end,
        float minFrameRms,
        out int speechSamples,
        out SpeakerTimelineFailure failure)
    {
        speechSamples = 0;
        if (end <= start)
        {
            failure = SpeakerTimelineFailure.RangeInvalid;
            return false;
        }

        lock (_lock)
        {
            failure = ClassifyLocked(streamEpoch, start, end);
            if (failure != SpeakerTimelineFailure.None)
                return false;

            int samplesPerFrame = Math.Max(1, _sampleRate / 100);
            double minEnergy = (double)minFrameRms * minFrameRms;
            long frameStart = start;
            while (frameStart < end)
            {
                int frameLength = (int)Math.Min(samplesPerFrame, end - frameStart);
                speechSamples += CountFrameSpeechSamples(frameStart, frameLength, minEnergy);
                frameStart += frameLength;
            }
            return true;
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _hasStream = false;
            _streamEpoch = 0;
            _retainedStart = 0;
            _segmentStart = 0;
            _end = 0;
            _gapGeneration++;
        }
    }

    private SpeakerTimelineFailure ClassifyLocked(long streamEpoch, long start, long end)
    {
        if (!_hasStream)
            return SpeakerTimelineFailure.NoStream;
        if (streamEpoch != _streamEpoch)
            return SpeakerTimelineFailure.StreamEpochMismatch;
        if (start < _retainedStart)
            return SpeakerTimelineFailure.AudioExpired;
        if (start < _segmentStart)
            return SpeakerTimelineFailure.TimelineGap;
        if (end > _end)
            return SpeakerTimelineFailure.NotYetAvailable;
        return SpeakerTimelineFailure.None;
    }

    private void MarkGapLocked()
    {
        if (_hasStream)
        {
            _segmentStart = _end;
            _gapGeneration++;
        }
    }

    private void CopyIn(ReadOnlySpan<float> audio, long startSample)
    {
        int capacity = _ring.Length;
        int offset = (int)(((startSample % capacity) + capacity) % capacity);
        int first = Math.Min(audio.Length, capacity - offset);
        audio.Slice(0, first).CopyTo(_ring.AsSpan(offset, first));
        if (first < audio.Length)
            audio.Slice(first).CopyTo(_ring.AsSpan(0, audio.Length - first));
    }

    private void CopyOut(Span<float> destination, long startSample)
    {
        int capacity = _ring.Length;
        int offset = (int)(((startSample % capacity) + capacity) % capacity);
        int first = Math.Min(destination.Length, capacity - offset);
        _ring.AsSpan(offset, first).CopyTo(destination.Slice(0, first));
        if (first < destination.Length)
            _ring.AsSpan(0, destination.Length - first).CopyTo(destination.Slice(first));
    }

    private int CountFrameSpeechSamples(long start, int length, double minEnergy)
    {
        int offset = (int)(start % _ring.Length);
        int firstLength = Math.Min(length, _ring.Length - offset);
        (double energy, int count) = MeasureEnergy(_ring.AsSpan(offset, firstLength), 0);
        (double totalEnergy, int remainingCount) = MeasureEnergy(_ring.AsSpan(0, length - firstLength), energy);
        int totalCount = count + remainingCount;
        return totalCount > 0 && totalEnergy / totalCount >= minEnergy ? totalCount : 0;
    }

    private static (double Energy, int Count) MeasureEnergy(ReadOnlySpan<float> samples, double energy)
    {
        int count = 0;
        foreach (float value in samples)
        {
            if (float.IsFinite(value))
            {
                energy += (double)value * value;
                count++;
            }
        }
        return (energy, count);
    }
}
