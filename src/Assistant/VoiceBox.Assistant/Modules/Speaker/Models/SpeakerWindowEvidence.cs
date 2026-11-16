namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Cached outcome of one hop-aligned window on the identity timeline.</summary>
internal sealed class SpeakerWindowEvidence
{
    private readonly Dictionary<SpeakerPurposeBank, SpeakerWindowScore> _scores = new();

    public SpeakerWindowEvidence(
        long streamEpoch,
        long startSample,
        long endSampleExclusive,
        SpeakerWindowEvidenceKind kind,
        SpeakerVerificationReason reason,
        float[]? embedding,
        SpeakerWindowQuality quality,
        bool isPrefix = false)
    {
        StreamEpoch = streamEpoch;
        StartSample = startSample;
        EndSampleExclusive = endSampleExclusive;
        Kind = kind;
        Reason = reason;
        Embedding = embedding;
        Quality = quality;
        IsPrefix = isPrefix;
    }

    public long StreamEpoch { get; }
    public long StartSample { get; }
    public long EndSampleExclusive { get; }
    public SpeakerWindowEvidenceKind Kind { get; }
    public SpeakerVerificationReason Reason { get; }
    public float[]? Embedding { get; }
    public SpeakerWindowQuality Quality { get; }
    public bool IsPrefix { get; }

    public void SetScore(SpeakerPurposeBank bank, SpeakerWindowScore score) => _scores[bank] = score;

    public bool TryGetScore(SpeakerPurposeBank bank, out SpeakerWindowScore score)
        => _scores.TryGetValue(bank, out score);
}
