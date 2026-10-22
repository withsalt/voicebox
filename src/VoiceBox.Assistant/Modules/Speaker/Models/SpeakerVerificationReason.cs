namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Typed reason attached to every speaker verdict; diagnostics never branch on strings.</summary>
internal enum SpeakerVerificationReason
{
    None,
    TimelineGap,
    StreamEpochMismatch,
    AudioExpired,
    CandidateRangeInvalid,
    MissingProfileVariant,
    MissingPurposeBank,
    InferenceOverload,
    InferenceTimeout,
    ProviderError,
    InvalidEmbedding,
    InsufficientSpeech,
    CancelledByNewTurn,
    RuntimeStopping,
    NotReady,
    CollectingCandidate,
    WaitingForEvidence,
    WaitingForInference,
    AmbiguousEvidence,
    HighConfidenceMatch,
    StrongUpstreamMatch,
    RecentOwnerPriorMatch,
    PrefixHighConfidenceMatch,
    MajorityMatch,
    MajorityMismatch,
}
