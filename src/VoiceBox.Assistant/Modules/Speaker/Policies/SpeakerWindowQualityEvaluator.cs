using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Policies;

/// <summary>
/// Assistant-side eligibility gate shared by enrollment, Wakeup, BargeInCommand and SingleTalk. A
/// window has to be finite, loud enough and mostly speech (by the energy mask) before the provider
/// is asked for an embedding. Eligibility is not identity: it only decides whether inference runs.
/// </summary>
internal sealed class SpeakerWindowQualityEvaluator
{
    public const double DefaultMinSpeechRatio = 0.5;
    public const double DefaultMaxClippingRatio = 0.002;
    public const double DefaultMaxDcOffset = 0.02;
    public const float ClippingLevel = 0.995f;

    private readonly float _minFrameRms;
    private readonly int _frameSamples;
    private readonly double _minSpeechRatio;
    private readonly double _maxClippingRatio;
    private readonly double _maxDcOffset;

    public SpeakerWindowQualityEvaluator(
        float minFrameRms,
        int sampleRate,
        double minSpeechRatio = DefaultMinSpeechRatio,
        double maxClippingRatio = DefaultMaxClippingRatio,
        double maxDcOffset = DefaultMaxDcOffset)
    {
        if (!float.IsFinite(minFrameRms) || minFrameRms < 0f)
            throw new ArgumentOutOfRangeException(nameof(minFrameRms));
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        _minFrameRms = minFrameRms;
        _frameSamples = Math.Max(1, sampleRate / 100);
        _minSpeechRatio = minSpeechRatio;
        _maxClippingRatio = maxClippingRatio;
        _maxDcOffset = maxDcOffset;
    }

    public float MinFrameRms => _minFrameRms;

    public SpeakerWindowQuality Evaluate(ReadOnlySpan<float> window)
    {
        if (window.IsEmpty)
            return SpeakerWindowQuality.Invalid;

        double energy = 0;
        double sum = 0;
        double peak = 0;
        int clipped = 0;
        for (int i = 0; i < window.Length; i++)
        {
            float value = window[i];
            if (!float.IsFinite(value))
                return SpeakerWindowQuality.Invalid;
            double abs = Math.Abs(value);
            if (abs > peak)
                peak = abs;
            if (abs >= ClippingLevel)
                clipped++;
            energy += (double)value * value;
            sum += value;
        }

        int speechSamples = CountSpeechSamples(window);
        return new SpeakerWindowQuality(
            AllFinite: true,
            Rms: Math.Sqrt(energy / window.Length),
            Peak: peak,
            ClippingRatio: clipped / (double)window.Length,
            DcOffset: Math.Abs(sum / window.Length),
            SpeechRatio: speechSamples / (double)window.Length,
            SpeechSamples: speechSamples,
            SampleCount: window.Length);
    }

    /// <summary>Samples that belong to 10ms frames whose RMS reaches the energy gate.</summary>
    public int CountSpeechSamples(ReadOnlySpan<float> audio)
    {
        double minEnergy = (double)_minFrameRms * _minFrameRms;
        int speech = 0;
        for (int start = 0; start < audio.Length; start += _frameSamples)
        {
            int length = Math.Min(_frameSamples, audio.Length - start);
            double energy = 0;
            for (int i = 0; i < length; i++)
            {
                float value = audio[start + i];
                if (float.IsFinite(value))
                    energy += (double)value * value;
            }
            if (energy / length >= minEnergy)
                speech += length;
        }
        return speech;
    }

    /// <summary>Whether the window may be handed to the embedding model at all.</summary>
    public bool IsEligible(in SpeakerWindowQuality quality)
        => quality.AllFinite &&
           quality.SampleCount > 0 &&
           quality.Rms >= _minFrameRms &&
           quality.SpeechRatio >= _minSpeechRatio &&
           quality.ClippingRatio <= _maxClippingRatio &&
           quality.DcOffset <= _maxDcOffset;

    /// <summary>Whether a single frame is loud enough to open or extend the speech gate.</summary>
    public bool IsSpeechFrame(ReadOnlySpan<float> audio)
    {
        if (audio.IsEmpty)
            return false;
        double energy = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            float value = audio[i];
            if (float.IsFinite(value))
                energy += (double)value * value;
        }
        return energy / audio.Length >= (double)_minFrameRms * _minFrameRms;
    }
}
