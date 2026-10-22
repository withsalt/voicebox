using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Profiles;

/// <summary>
/// Turns accepted enrollment segments into one prototype bank: pairwise similarity, medoid-based
/// outlier removal, an equal-weight centroid over the retained segments and representative prototypes
/// chosen by farthest-point coverage. Redundant directions do not fill unused prototype slots once
/// the minimum number of separate enrollment segments has been selected.
/// </summary>
internal sealed class SpeakerProfileBuilder
{
    // Only suppress numerically near-identical directions; this is not an acoustic clustering
    // threshold. Wider similarity cutoffs need calibration for the speaker provider.
    private const float DuplicateDirectionDistanceTolerance = 1e-6f;

    private readonly ISpeakerService _speaker;
    private readonly SpeakerEnrollmentOptions _options;

    public SpeakerProfileBuilder(ISpeakerService speaker, SpeakerEnrollmentOptions options)
    {
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Returns null when fewer than <paramref name="minRetained"/> segments survive the outlier rule.
    /// </summary>
    public SpeakerBankBuildResult? TryBuildBank(
        IReadOnlyList<SpeakerEnrollmentSegmentEmbedding> segments,
        int minRetained)
    {
        ArgumentNullException.ThrowIfNull(segments);
        if (segments.Count == 0)
            return null;

        int n = segments.Count;
        var matrix = new float[n, n];
        var pairwise = new List<float>();
        for (int i = 0; i < n; i++)
        {
            matrix[i, i] = 1f;
            for (int j = i + 1; j < n; j++)
            {
                float score = _speaker.CompareEmbeddings(segments[i].Embedding, segments[j].Embedding);
                matrix[i, j] = score;
                matrix[j, i] = score;
                pairwise.Add(score);
            }
        }

        // Each segment's agreement with the rest; the medoid is the most agreeable one.
        var agreement = new float[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < n; j++)
                if (j != i) sum += matrix[i, j];
            agreement[i] = n > 1 ? (float)(sum / (n - 1)) : 1f;
        }

        var retainedIndices = new List<int>();
        var discardedIndices = new List<int>();
        if (n >= 3)
        {
            double median = SpeakerEnrollmentQualityEvaluator.Median(agreement.ToList());
            double mad = SpeakerEnrollmentQualityEvaluator.Median(agreement.Select(v => (float)Math.Abs(v - median)).ToList());
            double cutoff = median - _options.OutlierMadMultiplier * Math.Max(mad, _options.OutlierMadFloor);
            for (int i = 0; i < n; i++)
            {
                if (agreement[i] < cutoff || agreement[i] < _speaker.Threshold)
                    discardedIndices.Add(i);
                else
                    retainedIndices.Add(i);
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
                retainedIndices.Add(i);
        }

        if (retainedIndices.Count < Math.Max(minRetained, SpeakerPrototypeBank.MinPrototypes))
        {
            return new SpeakerBankBuildResult(
                new SpeakerPrototypeBank([], []),
                retainedIndices.Select(i => segments[i]).ToArray(),
                discardedIndices.Select(i => segments[i]).ToArray(),
                SpeakerEnrollmentQualityEvaluator.Median(pairwise),
                Percentile(pairwise, 0.10),
                0);
        }

        var retained = retainedIndices.Select(i => segments[i]).ToArray();
        float[] centroid = SpeakerEmbeddingMath.NormalizedMean(retained.Select(static s => s.Embedding).ToArray());
        if (centroid.Length == 0)
            return null;

        // Farthest-point coverage: start with the medoid, then maximize distance to the nearest
        // selected prototype. The nearest prototype has the highest cosine similarity.
        int medoid = retainedIndices.OrderByDescending(i => agreement[i]).First();
        var selected = new List<int> { medoid };
        int maxPrototypes = Math.Clamp(_options.MaxPrototypesPerBank, SpeakerPrototypeBank.MinPrototypes, SpeakerPrototypeBank.MaxPrototypes);
        int minPrototypes = Math.Clamp(_options.MinPrototypesPerBank, SpeakerPrototypeBank.MinPrototypes, maxPrototypes);
        while (selected.Count < Math.Min(maxPrototypes, retainedIndices.Count))
        {
            int best = -1;
            float bestDistance = float.NegativeInfinity;
            foreach (int candidate in retainedIndices)
            {
                if (selected.Contains(candidate))
                    continue;
                float nearestSimilarity = selected.Max(s => matrix[candidate, s]);
                float distance = 1f - nearestSimilarity;
                if (distance > bestDistance)
                {
                    bestDistance = distance;
                    best = candidate;
                }
            }
            if (best < 0)
                break;
            // Keep the minimum number of distinct recordings for the matcher's top-2 rule even
            // when they share a direction. Beyond that, duplicates add no geometric coverage.
            if (selected.Count >= minPrototypes && bestDistance <= DuplicateDirectionDistanceTolerance)
                break;
            selected.Add(best);
        }

        var prototypes = selected
            .Select(i => new SpeakerPrototype(
                $"{segments[i].Condition.ToString().ToLowerInvariant()}-{segments[i].SegmentIndex + 1}",
                segments[i].Embedding,
                (float)Math.Clamp(agreement[i], 0f, 1f)))
            .ToList();

        double spread = 0;
        foreach (int i in retainedIndices)
            spread = Math.Max(spread, 1 - _speaker.CompareEmbeddings(segments[i].Embedding, centroid));

        return new SpeakerBankBuildResult(
            new SpeakerPrototypeBank(centroid, prototypes.AsReadOnly()),
            retained,
            discardedIndices.Select(i => segments[i]).ToArray(),
            SpeakerEnrollmentQualityEvaluator.Median(pairwise),
            Percentile(pairwise, 0.10),
            spread);
    }

    private static double Percentile(List<float> values, double percentile)
    {
        if (values.Count == 0)
            return 0;
        var sorted = values.ToArray();
        Array.Sort(sorted);
        double position = (sorted.Length - 1) * percentile;
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(sorted.Length - 1, lower + 1);
        double fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }
}
