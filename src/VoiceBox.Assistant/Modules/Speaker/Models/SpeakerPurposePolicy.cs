using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Validated, immutable decision parameters for one verification purpose.</summary>
internal sealed record SpeakerPurposePolicy(
    SpeakerVerificationPurpose Purpose,
    bool Enabled,
    float Threshold,
    float HighConfidenceThreshold,
    float IndeterminateScoreMargin,
    int WindowCount,
    int RequiredMatches,
    int MinimumSpeechSamples,
    TimeSpan InferenceTimeout,
    long RecentOwnerPriorSamples,
    bool AllowStrongUpstreamOneVote);
