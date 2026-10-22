using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Extensions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Policies;

/// <summary>
/// Immutable, validated speaker verification policy: provider geometry plus one
/// <see cref="SpeakerPurposePolicy"/> per purpose. It performs no I/O and holds no state.
/// </summary>
internal sealed class SpeakerVerificationPolicy
{
    private readonly SpeakerPurposePolicy _wakeup;
    private readonly SpeakerPurposePolicy _bargeInCommand;
    private readonly SpeakerPurposePolicy _singleTalk;

    private SpeakerVerificationPolicy(
        int sampleRate,
        int windowSamples,
        int hopSamples,
        long retentionSamples,
        int inferenceQueueCapacity,
        int prefetchMaxWindows,
        float minFrameRms,
        bool prefixHighConfidenceEnabled,
        int prefixHighConfidenceSamples,
        int wakeupPostTriggerEvidenceSamples,
        int wakeupStrongScoreMargin,
        SpeakerPurposePolicy wakeup,
        SpeakerPurposePolicy bargeInCommand,
        SpeakerPurposePolicy singleTalk)
    {
        SampleRate = sampleRate;
        WindowSamples = windowSamples;
        HopSamples = hopSamples;
        RetentionSamples = retentionSamples;
        InferenceQueueCapacity = inferenceQueueCapacity;
        PrefetchMaxWindows = prefetchMaxWindows;
        MinFrameRms = minFrameRms;
        PrefixHighConfidenceEnabled = prefixHighConfidenceEnabled;
        PrefixHighConfidenceSamples = prefixHighConfidenceSamples;
        WakeupPostTriggerEvidenceSamples = wakeupPostTriggerEvidenceSamples;
        WakeupStrongScoreMargin = wakeupStrongScoreMargin;
        _wakeup = wakeup;
        _bargeInCommand = bargeInCommand;
        _singleTalk = singleTalk;
    }

    public int SampleRate { get; }
    public int WindowSamples { get; }
    public int HopSamples { get; }
    public long RetentionSamples { get; }
    public int InferenceQueueCapacity { get; }
    public int PrefetchMaxWindows { get; }
    public float MinFrameRms { get; }
    public bool PrefixHighConfidenceEnabled { get; }
    public int PrefixHighConfidenceSamples { get; }
    public int WakeupPostTriggerEvidenceSamples { get; }
    public int WakeupStrongScoreMargin { get; }

    public SpeakerPurposePolicy Get(SpeakerVerificationPurpose purpose)
        => purpose switch
        {
            SpeakerVerificationPurpose.Wakeup => _wakeup,
            SpeakerVerificationPurpose.BargeInCommand => _bargeInCommand,
            SpeakerVerificationPurpose.SingleTalk => _singleTalk,
            _ => throw new ArgumentOutOfRangeException(nameof(purpose), purpose, null),
        };

    public float GetThreshold(SpeakerVerificationPurpose purpose) => Get(purpose).Threshold;

    public float GetHighConfidenceThreshold(SpeakerVerificationPurpose purpose) => Get(purpose).HighConfidenceThreshold;

    public int GetRequiredMatches(SpeakerVerificationPurpose purpose) => Get(purpose).RequiredMatches;

    public int GetWindowCount(SpeakerVerificationPurpose purpose) => Get(purpose).WindowCount;

    public IEnumerable<SpeakerPurposePolicy> EnabledPurposes
    {
        get
        {
            if (_wakeup.Enabled) yield return _wakeup;
            if (_bargeInCommand.Enabled) yield return _bargeInCommand;
            if (_singleTalk.Enabled) yield return _singleTalk;
        }
    }

    /// <summary>Banks every enabled purpose needs from the profile variant.</summary>
    public IReadOnlyCollection<SpeakerPurposeBank> RequiredBanks
        => EnabledPurposes.Select(static purpose => purpose.Purpose.Bank()).Distinct().ToArray();

    /// <summary>
    /// Builds the policy for the effective enabled purposes. <paramref name="effective"/> must
    /// already reflect the KWS gate: composition passes only purposes that are genuinely active.
    /// </summary>
    public static SpeakerVerificationPolicy Create(
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions options,
        int audioSampleRate,
        int audioChannels,
        SpeakerPurposeSet effective)
    {
        ArgumentNullException.ThrowIfNull(speaker);
        ArgumentNullException.ThrowIfNull(options);
        if (audioSampleRate != speaker.SampleRate || audioChannels != 1)
        {
            throw new InvalidOperationException(
                "Speaker identity input must be mono and match the speaker service: " +
                $"audio={audioSampleRate}Hz/{audioChannels}ch, speaker={speaker.SampleRate}Hz/1ch.");
        }

        int windowSamples = speaker.StreamingWindowSamples;
        int hopSamples = speaker.StreamingHopSamples;
        if (windowSamples <= 0 || hopSamples <= 0 || hopSamples > windowSamples)
        {
            throw new InvalidOperationException(
                "Speaker service returned invalid streaming window geometry: " +
                $"window={windowSamples}, hop={hopSamples} samples.");
        }

        int windowMs = checked((int)((long)windowSamples * 1000 / speaker.SampleRate));
        bool realtimePurposeEnabled = effective.Wakeup || effective.BargeInCommand;
        if (realtimePurposeEnabled && windowMs > options.MaxRealtimeWindowMs)
        {
            throw new InvalidOperationException(
                $"Speaker service window {windowMs}ms exceeds Speaker.MaxRealtimeWindowMs={options.MaxRealtimeWindowMs}; " +
                "it cannot serve realtime Wakeup/BargeInCommand verification.");
        }

        long retentionSamples = MillisecondsToSamples(speaker, options.AudioRetentionMs);
        if (retentionSamples < windowSamples * 2L)
            throw new InvalidOperationException("Speaker.AudioRetentionMs must cover at least two provider windows.");

        int prefixSamples = MillisecondsToSamples(speaker, options.PrefixHighConfidenceMs);
        if (options.PrefixHighConfidenceEnabled && prefixSamples >= windowSamples)
        {
            throw new InvalidOperationException(
                "Speaker.PrefixHighConfidenceMs must be smaller than the provider window when the prefix path is enabled.");
        }

        SpeakerConfirmationOptions confirmation = options.Confirmation
            ?? throw new InvalidOperationException("Speaker.Confirmation must not be null.");
        SpeakerPurposePolicy wakeup = CreatePurpose(
            SpeakerVerificationPurpose.Wakeup,
            effective.Wakeup,
            confirmation,
            options.Threshold,
            speaker);
        SpeakerPurposePolicy bargeIn = CreatePurpose(
            SpeakerVerificationPurpose.BargeInCommand,
            effective.BargeInCommand,
            confirmation,
            options.Threshold,
            speaker);
        SpeakerPurposePolicy singleTalk = CreatePurpose(
            SpeakerVerificationPurpose.SingleTalk,
            effective.SingleTalk,
            confirmation,
            options.Threshold,
            speaker);

        foreach (SpeakerPurposePolicy purpose in new[] { wakeup, bargeIn, singleTalk })
        {
            if (!purpose.Enabled)
                continue;
            long coverage = windowSamples + (long)(purpose.WindowCount - 1) * hopSamples;
            if (coverage > retentionSamples)
            {
                throw new InvalidOperationException(
                    $"Speaker.AudioRetentionMs must cover the {purpose.Purpose} decision window ({coverage} samples).");
            }
        }

        return new SpeakerVerificationPolicy(
            speaker.SampleRate,
            windowSamples,
            hopSamples,
            retentionSamples,
            options.InferenceQueueCapacity,
            options.PrefetchMaxWindows,
            options.MinFrameRms,
            options.PrefixHighConfidenceEnabled,
            prefixSamples,
            MillisecondsToSamples(speaker, options.WakeupPostTriggerEvidenceMs),
            options.WakeupStrongScoreMargin,
            wakeup,
            bargeIn,
            singleTalk);
    }

    private static SpeakerPurposePolicy CreatePurpose(
        SpeakerVerificationPurpose purpose,
        bool enabled,
        SpeakerConfirmationOptions options,
        float? fallbackThreshold,
        ISpeakerService speaker)
    {
        ArgumentNullException.ThrowIfNull(options);
        float threshold = options.Threshold ?? fallbackThreshold ?? speaker.Threshold;
        ValidateThreshold(threshold, $"{purpose} threshold");
        float highConfidence = options.HighConfidenceThreshold ?? Math.Min(1f, threshold + 0.15f);
        ValidateThreshold(highConfidence, $"{purpose} high-confidence threshold");
        if (options.HighConfidenceThreshold.HasValue && highConfidence <= threshold)
        {
            throw new InvalidOperationException(
                $"Speaker {purpose} high-confidence threshold must be greater than the effective normal threshold.");
        }
        if (!float.IsFinite(options.IndeterminateScoreMargin) || options.IndeterminateScoreMargin is < 0f or > 2f)
            throw new InvalidOperationException($"Speaker {purpose} indeterminate score margin must be within [0, 2].");
        if (options.WindowCount is <= 0 or > 64)
            throw new InvalidOperationException($"Speaker {purpose} WindowCount must be within [1, 64].");
        if (options.RequiredMatches <= 0 || options.RequiredMatches > options.WindowCount)
            throw new InvalidOperationException($"Speaker {purpose} RequiredMatches must be within [1, WindowCount].");
        if (options.MinimumSpeechMs is < 0 or > 60000)
            throw new InvalidOperationException($"Speaker {purpose} MinimumSpeechMs must be within [0, 60000].");
        if (options.InferenceTimeoutMs is < 0 or > 60000)
            throw new InvalidOperationException($"Speaker {purpose} InferenceTimeoutMs must be within [0, 60000].");
        if (options.RecentOwnerPriorMs is < 0 or > 60000)
            throw new InvalidOperationException($"Speaker {purpose} RecentOwnerPriorMs must be within [0, 60000].");

        return new SpeakerPurposePolicy(
            purpose,
            enabled,
            threshold,
            highConfidence,
            options.IndeterminateScoreMargin,
            options.WindowCount,
            options.RequiredMatches,
            MillisecondsToSamples(speaker, options.MinimumSpeechMs),
            TimeSpan.FromMilliseconds(options.InferenceTimeoutMs),
            MillisecondsToSamples(speaker, options.RecentOwnerPriorMs),
            options.AllowStrongUpstreamOneVote);
    }

    private static void ValidateThreshold(float threshold, string name)
    {
        if (!float.IsFinite(threshold) || threshold is < -1f or > 1f)
            throw new InvalidOperationException($"Speaker {name} {threshold} must be within [-1, 1].");
    }

    private static int MillisecondsToSamples(ISpeakerService speaker, int milliseconds)
        => checked((int)((long)speaker.SampleRate * milliseconds / 1000));
}
