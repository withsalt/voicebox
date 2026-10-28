using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Frozen near-end runtime configuration. The data plane never reads the mutable options object;
/// every budget is captured once at composition time together with its sample-rate dependent
/// limits, so a configuration reload cannot change the meaning of an in-flight capture.
/// </summary>
internal sealed class BargeInRuntimeOptions
{
    private BargeInRuntimeOptions(
        VoiceAssistantBargeInOptions options,
        int reservoirSamples,
        int preRollSamples,
        int timelineAlignedAudioLimitSamples,
        int maxCandidateSamples,
        int drainTimeoutSamples)
    {
        Enable = options.Enable;
        MinDurationMs = options.MinDurationMs;
        MinRms = options.MinRms;
        ActiveNearEndWindowMs = options.ActiveNearEndWindowMs;
        PlaybackMaxCandidateMs = options.PlaybackMaxCandidateMs;
        PlaybackRecognitionAudioLimitMs = options.PlaybackRecognitionAudioLimitMs;
        ActiveNearEndPreRollMs = options.ActiveNearEndPreRollMs;
        ActiveNearEndEndSilenceMs = options.ActiveNearEndEndSilenceMs;
        ActiveNearEndAudioDrainTimeoutMs = options.ActiveNearEndAudioDrainTimeoutMs;
        ActiveNearEndCooldownMs = options.ActiveNearEndCooldownMs;
        PlaybackDuckGain = options.PlaybackDuckGain;
        PlaybackDuckReleaseMs = options.PlaybackDuckReleaseMs;
        ReservoirSamples = reservoirSamples;
        PreRollSamples = preRollSamples;
        TimelineAlignedAudioLimitSamples = timelineAlignedAudioLimitSamples;
        MaxCandidateSamples = maxCandidateSamples;
        DrainTimeoutSamples = drainTimeoutSamples;
    }

    public bool Enable { get; }

    public int MinDurationMs { get; }

    public float MinRms { get; }

    public int ActiveNearEndWindowMs { get; }

    public int PlaybackMaxCandidateMs { get; }

    public int PlaybackRecognitionAudioLimitMs { get; }

    public int ActiveNearEndPreRollMs { get; }

    public int ActiveNearEndEndSilenceMs { get; }

    public int ActiveNearEndAudioDrainTimeoutMs { get; }

    public int ActiveNearEndCooldownMs { get; }

    public float PlaybackDuckGain { get; }

    public int PlaybackDuckReleaseMs { get; }

    public int ReservoirSamples { get; }

    public int PreRollSamples { get; }

    public int TimelineAlignedAudioLimitSamples { get; }

    public int MaxCandidateSamples { get; }

    public int DrainTimeoutSamples { get; }

    public static BargeInRuntimeOptions From(
        VoiceAssistantBargeInOptions options,
        AssistantAudioFormat audioFormat,
        bool identityRequired)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(audioFormat);

        int timelineAlignedAudioLimitMs = ComputeTimelineAlignedAudioLimitMs(options, identityRequired);
        int reservoirMs = Math.Max(timelineAlignedAudioLimitMs, options.ActiveNearEndWindowMs);
        return new BargeInRuntimeOptions(
            options,
            audioFormat.SamplesFromMillisecondsToInt32(
                reservoirMs,
                audioFormat.AudioSampleRate,
                audioFormat.AudioChannels),
            audioFormat.SamplesFromMillisecondsToInt32(
                options.ActiveNearEndPreRollMs,
                audioFormat.AudioSampleRate,
                audioFormat.AudioChannels),
            audioFormat.SamplesFromMillisecondsToInt32(
                timelineAlignedAudioLimitMs,
                audioFormat.AudioSampleRate,
                audioFormat.AudioChannels),
            audioFormat.SamplesFromMillisecondsToInt32(
                options.PlaybackMaxCandidateMs,
                audioFormat.AudioSampleRate,
                audioFormat.AudioChannels),
            audioFormat.SamplesFromMillisecondsToInt32(
                options.ActiveNearEndAudioDrainTimeoutMs,
                audioFormat.AudioSampleRate,
                audioFormat.AudioChannels));
    }

    private static int ComputeTimelineAlignedAudioLimitMs(
        VoiceAssistantBargeInOptions options,
        bool identityRequired)
    {
        // 两条异步处理链共享同一 capture timeline。复用已有的跨链 drain 预算作为
        // 对齐保留上限，不引入任何由当前设备实验得到的常数。
        int speakerCandidateAudioLimitMs = identityRequired
            ? checked(options.ActiveNearEndPreRollMs + options.PlaybackMaxCandidateMs)
            : options.ActiveNearEndPreRollMs;
        int candidateMs = Math.Max(
            speakerCandidateAudioLimitMs,
            Math.Max(
                checked(options.PlaybackMaxCandidateMs + options.ActiveNearEndPreRollMs),
                checked(options.ActiveNearEndWindowMs + options.ActiveNearEndPreRollMs)));
        return checked(candidateMs + options.ActiveNearEndAudioDrainTimeoutMs);
    }
}
