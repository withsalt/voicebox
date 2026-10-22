using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Enrollment;

internal static class SpeakerEmbeddingMath
{
    public static bool IsFiniteNonZero(ReadOnlySpan<float> embedding)
    {
        double sumSq = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            if (!float.IsFinite(embedding[i]))
                return false;
            sumSq += (double)embedding[i] * embedding[i];
        }
        return sumSq > 0;
    }

    /// <summary>Equal-weight mean, L2-normalized. Returns an empty array when the mean is degenerate.</summary>
    public static float[] NormalizedMean(IReadOnlyList<float[]> embeddings)
    {
        if (embeddings.Count == 0)
            return [];
        int size = embeddings[0].Length;
        var accumulated = new double[size];
        foreach (float[] embedding in embeddings)
        {
            if (embedding.Length != size)
                return [];
            for (int i = 0; i < size; i++)
                accumulated[i] += embedding[i];
        }
        double sumSq = 0;
        for (int i = 0; i < size; i++)
            sumSq += accumulated[i] * accumulated[i];
        if (!(sumSq > 0) || !double.IsFinite(sumSq))
            return [];
        double inverseNorm = 1d / Math.Sqrt(sumSq);
        var result = new float[size];
        for (int i = 0; i < size; i++)
            result[i] = (float)(accumulated[i] * inverseNorm);
        return result;
    }
}
