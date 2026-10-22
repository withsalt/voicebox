using VoiceBox.Audio.Common;
using VoiceBox.Audio.Common.AudioReader;
using VoiceBox.Audio.Common.Resampling;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed record WaveAudio(float[] Samples, int SampleRate, int SourceSampleRate, bool Resampled);

internal static class WaveAudioLoader
{
    public static WaveAudio LoadMono(string path, int targetSampleRate)
    {
        using var reader = new WaveStreamReader(path);
        int sourceRate = checked((int)reader.FormatChunk.SampleRate);
        int channels = reader.FormatChunk.NumChannels;
        if (sourceRate <= 0)
            throw new InvalidDataException($"Invalid sample rate in {path}.");
        if (channels <= 0)
            throw new InvalidDataException($"Invalid channel count in {path}.");

        int bytesPerSample = reader.FormatChunk.BitsPerSample / 8;
        if (bytesPerSample <= 0)
            throw new InvalidDataException($"Invalid bits per sample in {path}.");

        long interleavedCountLong = reader.RemainingBytes / bytesPerSample;
        if (interleavedCountLong > int.MaxValue)
            throw new InvalidDataException($"WAVE file is too large for this evaluator: {path}.");

        var interleaved = new float[(int)interleavedCountLong];
        int total = 0;
        while (total < interleaved.Length)
        {
            int read = reader.ReadSamples(interleaved.AsSpan(total));
            if (read == 0)
                break;
            total += read;
        }
        if (total != interleaved.Length)
            Array.Resize(ref interleaved, total);

        float[] mono = channels == 1 ? interleaved : Downmix(interleaved, channels);
        if (sourceRate == targetSampleRate)
            return new WaveAudio(mono, sourceRate, sourceRate, Resampled: false);

        using var resampler = new StreamingSpeexResampler(sourceRate, targetSampleRate, quality: 10);
        float[] main = [];
        int mainCount = resampler.Process(mono, ref main);
        float[] tail = [];
        int tailCount = resampler.Flush(ref tail);
        var output = new float[mainCount + tailCount];
        main.AsSpan(0, mainCount).CopyTo(output);
        tail.AsSpan(0, tailCount).CopyTo(output.AsSpan(mainCount));
        return new WaveAudio(output, targetSampleRate, sourceRate, Resampled: true);
    }

    private static float[] Downmix(float[] interleaved, int channels)
    {
        int frames = interleaved.Length / channels;
        var mono = new float[frames];
        for (int frame = 0; frame < frames; frame++)
        {
            double sum = 0;
            int offset = frame * channels;
            for (int channel = 0; channel < channels; channel++)
                sum += interleaved[offset + channel];
            mono[frame] = (float)(sum / channels);
        }
        return mono;
    }
}

internal static class OracleActivityAnalyzer
{
    private const int FrameMs = 20;
    private const int MaximumGapMs = 100;
    private const double AbsoluteFloorRms = 0.0005;
    private const double RelativeFloor = 0.03;

    public static OracleActivity Analyze(
        ReadOnlySpan<float> samples,
        double scale,
        int sampleRate,
        int startupGuardMs,
        int minimumSpeechMs)
    {
        if (samples.IsEmpty || sampleRate <= 0 || !double.IsFinite(scale) || scale <= 0)
            return OracleActivity.Unavailable;

        ComputeActivityFrames(
            samples,
            scale,
            sampleRate,
            out int frameSamples,
            out bool[] active,
            out double threshold);

        int minimumFrames = Math.Max(1, (minimumSpeechMs + FrameMs - 1) / FrameMs);
        // A frame may only become eligible once the complete startup guard has elapsed.
        // Round up so a guard that is not frame-aligned cannot admit an early frame.
        int guardFrame = startupGuardMs <= 0
            ? 0
            : (startupGuardMs - 1) / FrameMs + 1;
        int firstSpeech = -1;
        int eligible = -1;
        int lastSpeech = -1;
        int activeFrames = 0;

        for (int start = 0; start < active.Length;)
        {
            if (!active[start])
            {
                start++;
                continue;
            }

            int end = start + 1;
            while (end < active.Length && active[end])
                end++;

            if (firstSpeech < 0)
                firstSpeech = start;
            lastSpeech = end;
            activeFrames += end - start;

            int eligibleStart = Math.Max(start, guardFrame);
            if (eligible < 0 && end - eligibleStart >= minimumFrames)
                eligible = eligibleStart;

            start = end;
        }

        return new OracleActivity(
            Available: true,
            Eligible: eligible >= 0,
            FirstSpeechSample: firstSpeech >= 0 ? (long)firstSpeech * frameSamples : -1,
            EligibleOnsetSample: eligible >= 0 ? (long)eligible * frameSamples : -1,
            LastSpeechSampleExclusive: lastSpeech >= 0 ? Math.Min(samples.Length, (long)lastSpeech * frameSamples) : -1,
            ActiveDurationMs: activeFrames * FrameMs,
            ThresholdRms: threshold);
    }

    /// <summary>
    /// Active speech level in the sense of ITU-T P.56: the RMS of the signal restricted to the
    /// frames the activity detector considers speech, rather than the RMS of the whole file.
    /// <para>
    /// Whole-file RMS is not a usable level reference for a speech corpus. Two clips with the same
    /// perceived loudness but different amounts of leading/trailing/inter-word silence get different
    /// whole-file RMS values, so normalizing to whole-file RMS makes the level of the actual speech
    /// depend on how much silence the clip happens to contain. Every level ratio derived from it
    /// (SER, SNR) then varies per clip even when the dataset asks for a fixed value.
    /// </para>
    /// <para>
    /// The activity definition is shared with <see cref="Analyze"/> so that the onset reported for a
    /// track and the level it was normalized to are derived from one and the same segmentation.
    /// </para>
    /// </summary>
    public static double ActiveSpeechLevel(ReadOnlySpan<float> samples, int sampleRate)
    {
        if (samples.IsEmpty || sampleRate <= 0)
            return 0;

        ComputeActivityFrames(samples, scale: 1, sampleRate, out int frameSamples, out bool[] active, out _);

        double energy = 0;
        long counted = 0;
        for (int frame = 0; frame < active.Length; frame++)
        {
            if (!active[frame])
                continue;

            int start = frame * frameSamples;
            int count = Math.Min(frameSamples, samples.Length - start);
            for (int i = 0; i < count; i++)
            {
                double value = samples[start + i];
                energy += value * value;
            }
            counted += count;
        }
        return counted > 0 ? Math.Sqrt(energy / counted) : 0;
    }

    /// <summary>
    /// Splits the signal into <see cref="FrameMs"/> frames, marks the frames whose RMS clears a
    /// peak-relative threshold, and bridges short intra-word gaps. Shared by <see cref="Analyze"/>
    /// and <see cref="ActiveSpeechLevel"/> so onset and level cannot disagree about what "speech" is.
    /// </summary>
    private static void ComputeActivityFrames(
        ReadOnlySpan<float> samples,
        double scale,
        int sampleRate,
        out int frameSamples,
        out bool[] active,
        out double threshold)
    {
        frameSamples = Math.Max(1, sampleRate * FrameMs / 1000);
        int frameCount = (samples.Length + frameSamples - 1) / frameSamples;
        var rms = new double[frameCount];
        double maximum = 0;
        for (int frame = 0; frame < frameCount; frame++)
        {
            int start = frame * frameSamples;
            int count = Math.Min(frameSamples, samples.Length - start);
            double energy = 0;
            for (int i = 0; i < count; i++)
            {
                double value = samples[start + i] * scale;
                energy += value * value;
            }
            rms[frame] = count > 0 ? Math.Sqrt(energy / count) : 0;
            maximum = Math.Max(maximum, rms[frame]);
        }

        threshold = Math.Max(AbsoluteFloorRms, maximum * RelativeFloor);
        double localThreshold = threshold;
        active = rms.Select(value => value >= localThreshold).ToArray();
        FillShortGaps(active, Math.Max(1, MaximumGapMs / FrameMs));
    }

    private static void FillShortGaps(bool[] active, int maximumGapFrames)
    {
        int previousActive = -1;
        for (int i = 0; i < active.Length; i++)
        {
            if (!active[i])
                continue;

            if (previousActive >= 0)
            {
                int gap = i - previousActive - 1;
                if (gap is > 0 && gap <= maximumGapFrames)
                {
                    for (int j = previousActive + 1; j < i; j++)
                        active[j] = true;
                }
            }
            previousActive = i;
        }
    }
}
