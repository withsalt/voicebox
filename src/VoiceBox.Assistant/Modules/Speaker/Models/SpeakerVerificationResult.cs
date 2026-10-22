namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal readonly record struct SpeakerVerificationResult(
    SpeakerVerificationStatus Status,
    float Score,
    float Threshold,
    long StreamEpoch,
    long EvidenceStartSample,
    long EvidenceEndSampleExclusive,
    SpeakerVerificationReason Reason,
    int ObservedWindows = 0,
    int MatchedWindows = 0,
    int RequiredMatches = 0,
    bool HighConfidenceMatch = false,
    int AmbiguousWindows = 0,
    int RejectedWindows = 0,
    int InvalidWindows = 0,
    bool UsedStrongUpstreamEvidence = false,
    bool UsedRecentOwnerPrior = false,
    long RequiredEvidenceEndSampleExclusive = 0,
    float CentroidScore = 0,
    float BestPrototypeScore = 0,
    float Top2MeanScore = 0,
    SpeakerCandidateId CandidateId = default)
{
    public bool IsFinal => Status != SpeakerVerificationStatus.Pending;

    public bool IsDenied =>
        Status is SpeakerVerificationStatus.Mismatch
            or SpeakerVerificationStatus.InsufficientEvidence
            or SpeakerVerificationStatus.Unavailable
            or SpeakerVerificationStatus.Cancelled;

    public static SpeakerVerificationResult Pending(
        float threshold,
        in SpeakerEvidenceQuery query,
        SpeakerVerificationReason reason,
        int observedWindows = 0,
        int matchedWindows = 0,
        int requiredMatches = 0,
        int ambiguousWindows = 0,
        int rejectedWindows = 0,
        int invalidWindows = 0,
        long requiredEvidenceEndSampleExclusive = 0)
        => new(
            SpeakerVerificationStatus.Pending,
            0,
            threshold,
            query.StreamEpoch,
            query.CandidateStartSample,
            query.CandidateEndSampleExclusive,
            reason,
            observedWindows,
            matchedWindows,
            requiredMatches,
            AmbiguousWindows: ambiguousWindows,
            RejectedWindows: rejectedWindows,
            InvalidWindows: invalidWindows,
            RequiredEvidenceEndSampleExclusive: requiredEvidenceEndSampleExclusive,
            CandidateId: query.CandidateId);

    public static SpeakerVerificationResult Unavailable(
        float threshold,
        in SpeakerEvidenceQuery query,
        SpeakerVerificationReason reason,
        int requiredMatches = 0)
        => new(
            SpeakerVerificationStatus.Unavailable,
            0,
            threshold,
            query.StreamEpoch,
            query.CandidateStartSample,
            query.CandidateEndSampleExclusive,
            reason,
            RequiredMatches: requiredMatches,
            CandidateId: query.CandidateId);

    public static SpeakerVerificationResult InsufficientEvidence(
        float threshold,
        in SpeakerEvidenceQuery query,
        SpeakerVerificationReason reason,
        int observedWindows = 0,
        int matchedWindows = 0,
        int requiredMatches = 0,
        int ambiguousWindows = 0,
        int rejectedWindows = 0,
        int invalidWindows = 0)
        => new(
            SpeakerVerificationStatus.InsufficientEvidence,
            0,
            threshold,
            query.StreamEpoch,
            query.CandidateStartSample,
            query.CandidateEndSampleExclusive,
            reason,
            observedWindows,
            matchedWindows,
            requiredMatches,
            AmbiguousWindows: ambiguousWindows,
            RejectedWindows: rejectedWindows,
            InvalidWindows: invalidWindows,
            CandidateId: query.CandidateId);

    public static SpeakerVerificationResult Cancelled(
        float threshold,
        in SpeakerEvidenceQuery query,
        SpeakerVerificationReason reason)
        => new(
            SpeakerVerificationStatus.Cancelled,
            0,
            threshold,
            query.StreamEpoch,
            query.CandidateStartSample,
            query.CandidateEndSampleExclusive,
            reason,
            CandidateId: query.CandidateId);
}
