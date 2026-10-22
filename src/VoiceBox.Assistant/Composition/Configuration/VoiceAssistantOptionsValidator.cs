using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Composition.Configuration
{
    internal static class VoiceAssistantOptionsValidator
    {
        public static void Validate(VoiceAssistantOptions options, string optionsParameterName)
        {
            ArgumentNullException.ThrowIfNull(options);

            if (options.MinSentenceChars < 1)
                throw new ArgumentException("MinSentenceChars must be >= 1.", optionsParameterName);
            if (options.MaxHistoryTurns < 1)
                throw new ArgumentException("MaxHistoryTurns must be >= 1.", optionsParameterName);
            if (options.MaxResponseTokens is <= 0)
                throw new ArgumentException("MaxResponseTokens must be null or > 0.", optionsParameterName);
            if (options.PlaybackTailBufferMs < 0)
                throw new ArgumentException("PlaybackTailBufferMs must be >= 0.", optionsParameterName);
            if (options.PlaybackPrerollMs < 0)
                throw new ArgumentException("PlaybackPrerollMs must be >= 0.", optionsParameterName);
            if (options.Conversation is not { } conversation)
                throw new ArgumentException("Conversation must not be null.", optionsParameterName);
            if (conversation.ContinuousConversationIdleTimeout < TimeSpan.Zero)
                throw new ArgumentException("Conversation.ContinuousConversationIdleTimeout must be >= TimeSpan.Zero.", optionsParameterName);
            if (conversation.EndConversationKeywords is null)
                throw new ArgumentException("Conversation.EndConversationKeywords must not be null.", optionsParameterName);
            foreach (string? keyword in conversation.EndConversationKeywords)
            {
                if (string.IsNullOrWhiteSpace(keyword))
                {
                    throw new ArgumentException(
                        "Conversation.EndConversationKeywords cannot contain null or whitespace.",
                        optionsParameterName);
                }
            }
            if (options.Wakeup is not { } wakeup)
                throw new ArgumentException("Wakeup must not be null.", optionsParameterName);
            if (wakeup.PreRollMs < 0)
                throw new ArgumentException("Wakeup.PreRollMs must be >= 0.", optionsParameterName);
            if (wakeup.ReplayLeadInMs < 0)
                throw new ArgumentException("Wakeup.ReplayLeadInMs must be >= 0.", optionsParameterName);
            if (options.Speaker is not { } speaker)
                throw new ArgumentException("Speaker must not be null.", optionsParameterName);
            if (string.IsNullOrWhiteSpace(speaker.ProfilePath))
                throw new ArgumentException("Speaker.ProfilePath cannot be null or empty.", optionsParameterName);
            if (speaker.AudioRetentionMs is < 1000 or > 60000)
                throw new ArgumentException("Speaker.AudioRetentionMs must be within [1000, 60000].", optionsParameterName);
            if (speaker.InferenceQueueCapacity is < 1 or > 1024)
                throw new ArgumentException("Speaker.InferenceQueueCapacity must be within [1, 1024].", optionsParameterName);
            if (speaker.MaxRealtimeWindowMs is < 100 or > 60000)
                throw new ArgumentException("Speaker.MaxRealtimeWindowMs must be within [100, 60000].", optionsParameterName);
            if (speaker.PrefetchMaxWindows is < 1 or > 64)
                throw new ArgumentException("Speaker.PrefetchMaxWindows must be within [1, 64].", optionsParameterName);
            if (!float.IsFinite(speaker.MinFrameRms) || speaker.MinFrameRms is < 0f or > 1f)
                throw new ArgumentException("Speaker.MinFrameRms must be finite and within [0, 1].", optionsParameterName);
            if (speaker.PrefixHighConfidenceMs is < 0 or > 60000)
                throw new ArgumentException("Speaker.PrefixHighConfidenceMs must be within [0, 60000].", optionsParameterName);
            if (speaker.WakeupPostTriggerEvidenceMs is < 0 or > 60000)
                throw new ArgumentException("Speaker.WakeupPostTriggerEvidenceMs must be within [0, 60000].", optionsParameterName);
            if (speaker.WakeupStrongScoreMargin is < 0 or > 100)
                throw new ArgumentException("Speaker.WakeupStrongScoreMargin must be within [0, 100].", optionsParameterName);
            if (speaker.Threshold is { } speakerThreshold &&
                (!float.IsFinite(speakerThreshold) || speakerThreshold < -1f || speakerThreshold > 1f))
            {
                throw new ArgumentException("Speaker.Threshold must be null, finite, and within [-1, 1].", optionsParameterName);
            }
            ValidateConfirmation(speaker.Confirmation, speaker.Threshold, optionsParameterName);
            if (options.TtsSpeed is { } ttsSpeed && (!float.IsFinite(ttsSpeed) || ttsSpeed <= 0))
                throw new ArgumentException("TtsSpeed must be null, finite, and > 0.", optionsParameterName);
            if (options.Listening is not { } listening)
                throw new ArgumentException("Listening must not be null.", optionsParameterName);
            if (listening.MinDurationMs < 0)
                throw new ArgumentException("Listening.MinDurationMs must be >= 0.", optionsParameterName);
            if (!float.IsFinite(listening.MinRms) || listening.MinRms < 0)
                throw new ArgumentException("Listening.MinRms must be finite and >= 0.", optionsParameterName);

            if (options.BargeIn is not { } bargeIn)
                throw new ArgumentException("BargeIn must not be null.", optionsParameterName);
            if (bargeIn.MinDurationMs < 0)
                throw new ArgumentException("BargeIn.MinDurationMs must be >= 0.", optionsParameterName);
            if (!float.IsFinite(bargeIn.MinRms) || bargeIn.MinRms < 0)
                throw new ArgumentException("BargeIn.MinRms must be finite and >= 0.", optionsParameterName);
            if (bargeIn.PlaybackCandidateSuppressionMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.PlaybackCandidateSuppressionMs must be within [0, 60000].", optionsParameterName);
            if (bargeIn.PlaybackMaxCandidateMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.PlaybackMaxCandidateMs must be within [0, 60000].", optionsParameterName);
            if (speaker.Confirmation?.Enable == true && bargeIn.Enable)
            {
                if (bargeIn.PlaybackMaxCandidateMs <= 0)
                {
                    throw new ArgumentException(
                        "BargeIn.PlaybackMaxCandidateMs must be > 0 when Speaker.Confirmation.Enable and barge-in are both on.",
                        optionsParameterName);
                }
            }
            if (bargeIn.PlaybackRecognitionAudioLimitMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.PlaybackRecognitionAudioLimitMs must be within [0, 60000].", optionsParameterName);
            if (bargeIn.PlaybackStartupGuardMs < 0)
                throw new ArgumentException("BargeIn.PlaybackStartupGuardMs must be >= 0.", optionsParameterName);
            if (!float.IsFinite(bargeIn.PlaybackDuckGain) || bargeIn.PlaybackDuckGain is < 0f or > 1f)
                throw new ArgumentException("BargeIn.PlaybackDuckGain must be finite and within [0, 1].", optionsParameterName);
            if (bargeIn.PlaybackDuckReleaseMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.PlaybackDuckReleaseMs must be within [0, 60000].", optionsParameterName);
            if (bargeIn.ActiveNearEndWindowMs is <= 0 or > 60000)
                throw new ArgumentException("BargeIn.ActiveNearEndWindowMs must be within [1, 60000].", optionsParameterName);
            if (bargeIn.ActiveNearEndPreRollMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.ActiveNearEndPreRollMs must be within [0, 60000].", optionsParameterName);
            if (bargeIn.ActiveNearEndEndSilenceMs <= 0)
                throw new ArgumentException("BargeIn.ActiveNearEndEndSilenceMs must be > 0.", optionsParameterName);
            if (bargeIn.ActiveNearEndAudioDrainTimeoutMs is <= 0 or > 60000)
                throw new ArgumentException("BargeIn.ActiveNearEndAudioDrainTimeoutMs must be within [1, 60000].", optionsParameterName);
            if (bargeIn.ActiveNearEndCooldownMs < 0)
                throw new ArgumentException("BargeIn.ActiveNearEndCooldownMs must be >= 0.", optionsParameterName);
            if (bargeIn.Dtd is not { } dtd)
                throw new ArgumentException("BargeIn.Dtd must not be null.", optionsParameterName);
            if (dtd.Model is null)
                throw new ArgumentException("BargeIn.Dtd.Model must not be null.", optionsParameterName);
            if (dtd.DecisionHistoryMs is < 100 or > 60000)
                throw new ArgumentException("BargeIn.Dtd.DecisionHistoryMs must be within [100, 60000].", optionsParameterName);
            if (dtd.AcousticPathChangeGuardMs is < 0 or > 60000)
                throw new ArgumentException("BargeIn.Dtd.AcousticPathChangeGuardMs must be within [0, 60000].", optionsParameterName);
            if (bargeIn.RenderAlignment is not { } alignment)
                throw new ArgumentException("BargeIn.RenderAlignment must not be null.", optionsParameterName);
            if (alignment.DelayHintMaxAgeMs is < 1 or > 60000)
                throw new ArgumentException("BargeIn.RenderAlignment.DelayHintMaxAgeMs must be within [1, 60000].", optionsParameterName);
            if (alignment.EstimateHoldoverMs is < 1 or > 60000)
                throw new ArgumentException("BargeIn.RenderAlignment.EstimateHoldoverMs must be within [1, 60000].", optionsParameterName);
            if (alignment.MaxTrackedDelayMs is < 0 or > 5000)
                throw new ArgumentException("BargeIn.RenderAlignment.MaxTrackedDelayMs must be within [0, 5000].", optionsParameterName);
            if (alignment.PathChangeThresholdMs < 0 || alignment.PathChangeThresholdMs > alignment.MaxTrackedDelayMs)
                throw new ArgumentException("BargeIn.RenderAlignment.PathChangeThresholdMs must be within [0, MaxTrackedDelayMs].", optionsParameterName);
            if (alignment.DelayUpdateIntervalMs is < 1 or > 60000)
                throw new ArgumentException("BargeIn.RenderAlignment.DelayUpdateIntervalMs must be within [1, 60000].", optionsParameterName);
            if (!double.IsFinite(alignment.MaximumDriftPpm) || alignment.MaximumDriftPpm < 0 || alignment.MaximumDriftPpm > 10000)
                throw new ArgumentException("BargeIn.RenderAlignment.MaximumDriftPpm must be finite and within [0, 10000].", optionsParameterName);
            if (!double.IsFinite(alignment.MaximumDelaySlewPerSecond) || alignment.MaximumDelaySlewPerSecond <= 0 || alignment.MaximumDelaySlewPerSecond > 16000)
                throw new ArgumentException("BargeIn.RenderAlignment.MaximumDelaySlewPerSecond must be finite and within (0, 16000].", optionsParameterName);
            if (!double.IsFinite(alignment.MinimumAlignmentConfidence) || alignment.MinimumAlignmentConfidence < 0 || alignment.MinimumAlignmentConfidence > 1)
                throw new ArgumentException("BargeIn.RenderAlignment.MinimumAlignmentConfidence must be finite and within [0, 1].", optionsParameterName);

            if (options.Tools is not { } tools)
                throw new ArgumentException("Tools must not be null.", optionsParameterName);
            if (tools.ToolTranscriptRetentionTurns < 0)
                throw new ArgumentException("Tools.ToolTranscriptRetentionTurns must be >= 0.", optionsParameterName);
            if (tools.MaxPersistedToolResultBytes < 256)
                throw new ArgumentException("Tools.MaxPersistedToolResultBytes must be >= 256.", optionsParameterName);
            if (tools.MaxParallelToolCalls is < 1 or > 64)
                throw new ArgumentException("Tools.MaxParallelToolCalls must be within [1, 64].", optionsParameterName);
            if (tools.MaxToolRetryAttempts is < 0 or > 8)
                throw new ArgumentException("Tools.MaxToolRetryAttempts must be within [0, 8].", optionsParameterName);
            if (tools.ToolRetryBaseDelayMs is < 0 or > 60000)
                throw new ArgumentException("Tools.ToolRetryBaseDelayMs must be within [0, 60000].", optionsParameterName);
            if (tools.MaxToolRetryBudgetMs < 1)
                throw new ArgumentException("Tools.MaxToolRetryBudgetMs must be >= 1.", optionsParameterName);
        }

        private static void ValidateConfirmation(
            SpeakerConfirmationOptions confirmation,
            float? fallbackThreshold,
            string optionsParameterName)
        {
            const string name = "Speaker.Confirmation";
            if (confirmation is null)
                throw new ArgumentException($"{name} must not be null.", optionsParameterName);
            if (confirmation.MinimumSpeechMs is < 0 or > 60000)
                throw new ArgumentException($"{name}.MinimumSpeechMs must be within [0, 60000].", optionsParameterName);
            if (confirmation.WindowCount is <= 0 or > 64)
                throw new ArgumentException($"{name}.WindowCount must be within [1, 64].", optionsParameterName);
            if (confirmation.RequiredMatches <= 0 || confirmation.RequiredMatches > confirmation.WindowCount)
                throw new ArgumentException($"{name}.RequiredMatches must be within [1, WindowCount].", optionsParameterName);
            if (confirmation.InferenceTimeoutMs is < 0 or > 60000)
                throw new ArgumentException($"{name}.InferenceTimeoutMs must be within [0, 60000].", optionsParameterName);
            if (confirmation.RecentOwnerPriorMs is < 0 or > 60000)
                throw new ArgumentException($"{name}.RecentOwnerPriorMs must be within [0, 60000].", optionsParameterName);
            if (!float.IsFinite(confirmation.IndeterminateScoreMargin) || confirmation.IndeterminateScoreMargin is < 0f or > 2f)
                throw new ArgumentException($"{name}.IndeterminateScoreMargin must be finite and within [0, 2].", optionsParameterName);
            if (confirmation.Threshold is { } threshold && (!float.IsFinite(threshold) || threshold < -1f || threshold > 1f))
                throw new ArgumentException($"{name}.Threshold must be null, finite, and within [-1, 1].", optionsParameterName);
            if (confirmation.HighConfidenceThreshold is { } high && (!float.IsFinite(high) || high < -1f || high > 1f))
                throw new ArgumentException($"{name}.HighConfidenceThreshold must be null, finite, and within [-1, 1].", optionsParameterName);
            float? effective = confirmation.Threshold ?? fallbackThreshold;
            if (effective is { } normal && confirmation.HighConfidenceThreshold is { } highThreshold && highThreshold <= normal)
            {
                throw new ArgumentException(
                    $"{name}.HighConfidenceThreshold must be greater than the effective normal threshold.",
                    optionsParameterName);
            }
        }

        /// <summary>
        /// Validates speaker decision coverage against the live <see cref="ISpeakerService"/>
        /// streaming window/hop for the purposes that are effectively enabled. Call after
        /// <see cref="Validate"/> whenever an online speaker module will be created.
        /// </summary>
        public static void ValidateSpeakerGeometry(
            VoiceAssistantOptions options,
            ISpeakerService speakerService,
            SpeakerPurposeSet effective,
            string optionsParameterName)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(speakerService);

            VoiceAssistantSpeakerOptions speaker = options.Speaker
                ?? throw new ArgumentException("Speaker must not be null.", optionsParameterName);
            VoiceAssistantWakeupOptions wakeup = options.Wakeup
                ?? throw new ArgumentException("Wakeup must not be null.", optionsParameterName);
            VoiceAssistantBargeInOptions bargeIn = options.BargeIn
                ?? throw new ArgumentException("BargeIn must not be null.", optionsParameterName);

            int windowSamples = speakerService.StreamingWindowSamples;
            int hopSamples = speakerService.StreamingHopSamples;
            if (windowSamples <= 0 || hopSamples <= 0 || hopSamples > windowSamples)
            {
                throw new ArgumentException(
                    "Speaker service returned invalid streaming window geometry: " +
                    $"window={windowSamples}, hop={hopSamples} samples.",
                    optionsParameterName);
            }

            int windowMs = SamplesToMilliseconds(windowSamples, speakerService.SampleRate);
            int hopMs = SamplesToMilliseconds(hopSamples, speakerService.SampleRate);
            if (windowMs <= 0 || hopMs <= 0)
            {
                throw new ArgumentException(
                    "Speaker streaming window/hop is not representable in milliseconds.",
                    optionsParameterName);
            }

            if ((effective.Wakeup || effective.BargeInCommand) && windowMs > speaker.MaxRealtimeWindowMs)
            {
                throw new ArgumentException(
                    $"Speaker service window {windowMs}ms exceeds Speaker.MaxRealtimeWindowMs={speaker.MaxRealtimeWindowMs}; " +
                    "it cannot serve realtime Wakeup/BargeInCommand verification.",
                    optionsParameterName);
            }
            if (speaker.PrefixHighConfidenceEnabled && speaker.PrefixHighConfidenceMs >= windowMs)
            {
                throw new ArgumentException(
                    "Speaker.PrefixHighConfidenceMs must be smaller than the speaker service window when enabled.",
                    optionsParameterName);
            }

            SpeakerConfirmationOptions confirmation = speaker.Confirmation
                ?? throw new ArgumentException("Speaker.Confirmation must not be null.", optionsParameterName);
            long maxCoverageMs = 0;
            if (effective.Any)
            {
                long coverageMs = Math.Max(
                    confirmation.MinimumSpeechMs,
                    (long)windowMs + (long)(confirmation.WindowCount - 1) * hopMs);
                if (coverageMs > 60000)
                    throw new ArgumentException("Speaker.Confirmation window coverage must be <= 60000ms.", optionsParameterName);
                if (ComputeWindowSlots(coverageMs, windowMs, hopMs) > 64)
                    throw new ArgumentException("Speaker.Confirmation decision coverage must require no more than 64 windows.", optionsParameterName);
                maxCoverageMs = coverageMs;
            }

            if (speaker.AudioRetentionMs < maxCoverageMs + (effective.Wakeup ? wakeup.PreRollMs : 0))
            {
                throw new ArgumentException(
                    "Speaker.AudioRetentionMs must cover the largest decision window plus the wakeup pre-roll.",
                    optionsParameterName);
            }
            if (effective.Wakeup && wakeup.PreRollMs < windowMs)
            {
                throw new ArgumentException(
                    "Wakeup.PreRollMs must be >= the speaker service window when owner-speaker verification is enabled.",
                    optionsParameterName);
            }
            if (effective.BargeInCommand && bargeIn.Enable)
            {
                long bargeInCoverageMs = (long)windowMs + (long)(confirmation.WindowCount - 1) * hopMs;
                if (bargeIn.PlaybackMaxCandidateMs < bargeInCoverageMs)
                {
                    throw new ArgumentException(
                        $"BargeIn.PlaybackMaxCandidateMs must be >= {bargeInCoverageMs} " +
                        "when owner-speaker barge-in verification is enabled.",
                        optionsParameterName);
                }
                if (ComputeWindowSlots(bargeIn.PlaybackMaxCandidateMs, windowMs, hopMs) > 64)
                {
                    throw new ArgumentException(
                        "Owner-speaker barge-in candidate must contain no more than 64 windows.",
                        optionsParameterName);
                }
            }
        }

        private static int SamplesToMilliseconds(int samples, int sampleRate)
        {
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));
            return checked((int)((long)samples * 1000 / sampleRate));
        }

        private static long ComputeWindowSlots(long audioMs, int windowMs, int hopMs)
        {
            if (audioMs < windowMs)
                return 0;
            return (audioMs - windowMs) / hopMs + 1;
        }
    }
}
