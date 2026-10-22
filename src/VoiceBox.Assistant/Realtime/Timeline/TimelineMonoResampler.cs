using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Realtime.Timeline;

/// <summary>
/// Converts a source-timeline signal to mono 16 kHz samples without losing the
/// source position represented by buffered processor output. The destination is
/// bounded by the caller, so this class only owns interpolation state.
/// </summary>
internal sealed class TimelineMonoResampler
{
    private uint _sampleRate;
    private uint _channels;
    private long _streamEpoch;
    private long _originSample;
    private long _expectedSourceFrame;
    private long _nextTargetSample;
    private float _previous;
    private bool _configured;
    private bool _hasPrevious;

    public void Append(
        in CaptureSignalTimeline timeline,
        ReadOnlyMemory<float> buffer,
        uint sampleRate,
        uint channels,
        long originSample,
        long streamEpoch,
        BoundedFractionalRenderBuffer destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (!timeline.IsValid || timeline.SampleCount <= 0)
            return;
        if (sampleRate == 0 || channels == 0 || originSample < 0 || streamEpoch <= 0)
            throw new ArgumentOutOfRangeException(nameof(timeline), "Audio timeline format is invalid.");
        if (timeline.SourceRange.StreamEpoch != streamEpoch ||
            timeline.SourceRange.StartSample % channels != 0 ||
            timeline.SourceRange.EndSampleExclusive % channels != 0 ||
            timeline.SampleCount % channels != 0)
        {
            throw new InvalidOperationException("Buffered audio timeline is not channel aligned with the stream.");
        }

        ReadOnlyMemory<float> valid = timeline.Slice(buffer);
        if (valid.IsEmpty)
            return;

        long sourceStartFrame = timeline.SourceRange.StartSample / channels;
        long sourceEndFrame = timeline.SourceRange.EndSampleExclusive / channels;
        long originFrame = originSample / channels;
        if (sourceEndFrame <= sourceStartFrame || originSample % channels != 0)
            throw new InvalidOperationException("Buffered audio timeline has an invalid frame range.");

        if (!_configured)
        {
            _sampleRate = sampleRate;
            _channels = channels;
            _streamEpoch = streamEpoch;
            _originSample = originSample;
            _expectedSourceFrame = sourceStartFrame;
            // A buffered capture processor can still emit source audio from before a
            // newly-established detector origin. That prefix belongs to the discarded
            // stream state and must not be represented by negative target positions.
            _nextTargetSample = Math.Max(
                0,
                checked((long)Math.Ceiling(
                    (sourceStartFrame - originFrame) * 16000.0 / sampleRate)));
            _configured = true;
        }
        else if (_sampleRate != sampleRate ||
                 _channels != channels ||
                 _streamEpoch != streamEpoch ||
                 _originSample != originSample ||
                 sourceStartFrame != _expectedSourceFrame)
        {
            throw new InvalidOperationException("Buffered audio timeline is discontinuous.");
        }

        double sourceStep = sampleRate / 16000.0;
        while (true)
        {
            double sourcePosition = originFrame + _nextTargetSample * sourceStep;
            if (sourcePosition + 1.0 >= sourceEndFrame)
                break;

            long firstFrame = (long)Math.Floor(sourcePosition);
            double fraction = sourcePosition - firstFrame;
            float first = ReadMono(valid.Span, firstFrame, sourceStartFrame, channels);
            float second = ReadMono(valid.Span, checked(firstFrame + 1), sourceStartFrame, channels);
            destination.Append(
                first + (second - first) * (float)fraction,
                _nextTargetSample);
            _nextTargetSample = checked(_nextTargetSample + 1);
        }

        _previous = ReadMono(valid.Span, sourceEndFrame - 1, sourceStartFrame, channels);
        _hasPrevious = true;
        _expectedSourceFrame = sourceEndFrame;
    }

    public void Reset()
    {
        _sampleRate = 0;
        _channels = 0;
        _streamEpoch = 0;
        _originSample = 0;
        _expectedSourceFrame = 0;
        _nextTargetSample = 0;
        _previous = 0;
        _configured = false;
        _hasPrevious = false;
    }

    private float ReadMono(
        ReadOnlySpan<float> samples,
        long frame,
        long sourceStartFrame,
        uint channels)
    {
        if (frame == sourceStartFrame - 1 && _hasPrevious)
            return _previous;
        long relativeFrame = frame - sourceStartFrame;
        if (relativeFrame < 0 || relativeFrame >= samples.Length / channels)
            throw new InvalidOperationException("Render-aware resampler requested a sample outside the timeline.");

        int offset = checked((int)(relativeFrame * channels));
        // Mono skips the per-channel loop and the divide by exactly 1.0, which is an
        // exact operation, so the returned bits are unchanged.
        if (channels == 1)
        {
            float mono = samples[offset];
            return float.IsFinite(mono) ? mono : 0f;
        }

        double sum = 0;
        for (int channel = 0; channel < channels; channel++)
        {
            float sample = samples[offset + channel];
            if (float.IsFinite(sample))
                sum += sample;
        }
        return (float)(sum / channels);
    }
}
