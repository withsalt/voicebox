using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Enrollment;

/// <summary>
/// Per-segment quality gate for enrollment and confirmation. It reuses the online window quality
/// evaluator so a segment can only contribute windows that online verification would also have
/// inferred, then reduces the accepted windows to one normalized segment embedding.
/// </summary>
internal sealed class SpeakerEnrollmentQualityEvaluator
{
    private readonly ISpeakerService _speaker;
    private readonly SpeakerEnrollmentOptions _options;
    private readonly SpeakerWindowQualityEvaluator _windows;

    public SpeakerEnrollmentQualityEvaluator(ISpeakerService speaker, SpeakerEnrollmentOptions options)
    {
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _windows = new SpeakerWindowQualityEvaluator(
            options.MinimumRms,
            speaker.SampleRate,
            minSpeechRatio: 0.5,
            options.MaxClippingRatio,
            options.MaxDcOffset);
    }

    public SpeakerEnrollmentSegmentEvaluation Evaluate(
        SpeakerEnrollmentSegmentCapture capture,
        SpeakerEnrollmentCondition condition,
        int segmentIndex,
        int requiredSpeechSamples,
        bool nearEndGateAvailable)
    {
        ArgumentNullException.ThrowIfNull(capture);
        float[] speech = capture.SpeechAudio;
        SpeakerWindowQuality whole = _windows.Evaluate(speech);
        double speechRatio = capture.TotalSamples > 0 ? Math.Min(1.0, speech.Length / (double)capture.TotalSamples) : 0;
        double renderRatio = capture.FrameCount > 0 ? capture.RenderActiveFrames / (double)capture.FrameCount : 0;
        double nearEndRatio = capture.FrameCount > 0 ? capture.NearEndAcceptedFrames / (double)capture.FrameCount : 0;
        var quality = new SpeakerEnrollmentSegmentQuality(
            TimeSpan.FromSeconds(speech.Length / (double)_speaker.SampleRate),
            whole.Rms,
            whole.Peak,
            whole.ClippingRatio,
            whole.DcOffset,
            speechRatio,
            renderRatio,
            nearEndRatio,
            0,
            0,
            0);

        if (speech.Length < requiredSpeechSamples)
            return new(SpeakerEnrollmentReasonCode.InsufficientSpeech, quality, null);
        if (!whole.AllFinite)
            return new(SpeakerEnrollmentReasonCode.InvalidEmbedding, quality, null);
        if (whole.Rms < _options.MinimumRms)
            return new(SpeakerEnrollmentReasonCode.LowEnergy, quality, null);
        if (whole.ClippingRatio > _options.MaxClippingRatio)
            return new(SpeakerEnrollmentReasonCode.Clipping, quality, null);
        if (whole.DcOffset > _options.MaxDcOffset)
            return new(SpeakerEnrollmentReasonCode.DcOffset, quality, null);
        if (speechRatio < _options.MinSpeechRatio || speechRatio > _options.MaxSpeechRatio)
            return new(SpeakerEnrollmentReasonCode.InsufficientSpeech, quality, null);
        if (condition == SpeakerEnrollmentCondition.DoubleTalk)
        {
            if (renderRatio < _options.MinRenderActiveRatio)
                return new(SpeakerEnrollmentReasonCode.NoRenderDuringDoubleTalk, quality, null);
            if (nearEndGateAvailable && capture.NearEndObserved && nearEndRatio < _options.MinNearEndAcceptedRatio)
                return new(SpeakerEnrollmentReasonCode.InsufficientNearEndDuringDoubleTalk, quality, null);
        }

        List<float[]> windowEmbeddings = ExtractWindowEmbeddings(speech);
        if (windowEmbeddings.Count == 0)
            return new(SpeakerEnrollmentReasonCode.InvalidEmbedding, quality, null);

        double withinMedian = windowEmbeddings.Count < 2
            ? 1.0
            : Median(PairwiseScores(windowEmbeddings));
        quality = quality with
        {
            EmbeddingWindowCount = windowEmbeddings.Count,
            WithinSegmentSimilarityMedian = withinMedian,
        };
        if (windowEmbeddings.Count >= 2 && withinMedian < _options.MinWithinSegmentSimilarity)
            return new(SpeakerEnrollmentReasonCode.LowWithinSegmentConsistency, quality, null);

        float[] segmentEmbedding = SpeakerEmbeddingMath.NormalizedMean(windowEmbeddings);
        if (segmentEmbedding.Length == 0)
            return new(SpeakerEnrollmentReasonCode.InvalidEmbedding, quality, null);

        return new(
            SpeakerEnrollmentReasonCode.SegmentAccepted,
            quality,
            new SpeakerEnrollmentSegmentEmbedding(condition, segmentIndex, segmentEmbedding, windowEmbeddings.ToArray(), quality));
    }

    /// <summary>Hop-aligned provider windows of the segment that pass the online window gate.</summary>
    public List<float[]> ExtractWindowEmbeddings(ReadOnlySpan<float> speech)
    {
        int window = _speaker.StreamingWindowSamples;
        int hop = Math.Max(1, _speaker.StreamingHopSamples);
        var embeddings = new List<float[]>();
        if (window <= 0)
            return embeddings;
        if (speech.Length < window)
        {
            SpeakerWindowQuality quality = _windows.Evaluate(speech);
            if (_windows.IsEligible(in quality))
                TryAdd(embeddings, speech);
            return embeddings;
        }

        for (int start = 0; start + window <= speech.Length; start += hop)
        {
            ReadOnlySpan<float> slice = speech.Slice(start, window);
            SpeakerWindowQuality quality = _windows.Evaluate(slice);
            if (_windows.IsEligible(in quality))
                TryAdd(embeddings, slice);
        }
        return embeddings;
    }

    private void TryAdd(List<float[]> embeddings, ReadOnlySpan<float> audio)
    {
        float[] embedding = _speaker.ExtractEmbedding(audio);
        if (embedding is null || embedding.Length != _speaker.EmbeddingSize)
            return;
        if (SpeakerEmbeddingMath.IsFiniteNonZero(embedding))
            embeddings.Add(embedding);
    }

    public List<float> PairwiseScores(IReadOnlyList<float[]> embeddings)
    {
        var scores = new List<float>();
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = i + 1; j < embeddings.Count; j++)
                scores.Add(_speaker.CompareEmbeddings(embeddings[i], embeddings[j]));
        }
        return scores;
    }

    public static double Median(List<float> values)
    {
        if (values.Count == 0)
            return 0;
        var sorted = values.ToArray();
        Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[middle - 1] + sorted[middle]) / 2.0
            : sorted[middle];
    }
}
