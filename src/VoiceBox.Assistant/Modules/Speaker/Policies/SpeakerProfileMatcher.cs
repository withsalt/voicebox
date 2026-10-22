using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Policies;

/// <summary>
/// Multi-prototype identity rule. A window matches when either the two closest prototypes agree
/// (<c>top2Mean &gt;= threshold</c>) or the centroid and the best prototype both agree; a single
/// outlying prototype can never authorize on its own. High confidence additionally requires the
/// top-2 agreement to clear the high threshold while the centroid still supports it.
/// </summary>
internal static class SpeakerProfileMatcher
{
    public static SpeakerWindowScore Score(
        ISpeakerService speaker,
        SpeakerPrototypeBank bank,
        ReadOnlySpan<float> embedding)
    {
        ArgumentNullException.ThrowIfNull(speaker);
        ArgumentNullException.ThrowIfNull(bank);
        if (!bank.IsUsable)
            throw new InvalidOperationException("Speaker prototype bank needs at least two prototypes and a centroid.");

        float centroid = speaker.CompareEmbeddings(embedding, bank.Centroid);
        float best = float.NegativeInfinity;
        float second = float.NegativeInfinity;
        foreach (SpeakerPrototype prototype in bank.Prototypes)
        {
            float score = speaker.CompareEmbeddings(embedding, prototype.Embedding);
            if (score > best)
            {
                second = best;
                best = score;
            }
            else if (score > second)
            {
                second = score;
            }
        }

        float top2 = (best + second) / 2f;
        return new SpeakerWindowScore(centroid, best, top2);
    }

    /// <summary>Threshold comparisons use <c>&gt;=</c> everywhere; boundary tests pin this.</summary>
    public static bool IsMatch(in SpeakerWindowScore score, float threshold)
        => score.IsFinite &&
           (score.Top2MeanScore >= threshold ||
            (score.CentroidScore >= threshold && score.BestPrototypeScore >= threshold));

    public static bool IsHighConfidence(in SpeakerWindowScore score, float threshold, float highConfidenceThreshold)
        => score.IsFinite &&
           highConfidenceThreshold > threshold &&
           score.Top2MeanScore >= highConfidenceThreshold &&
           score.CentroidScore >= threshold;

    /// <summary>Clearly below threshold: not even the indeterminate margin is reached.</summary>
    public static bool IsRejected(in SpeakerWindowScore score, float threshold, float indeterminateMargin)
        => score.IsFinite && score.Reported < threshold - indeterminateMargin;
}
