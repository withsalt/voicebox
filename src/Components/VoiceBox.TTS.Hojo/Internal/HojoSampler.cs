namespace VoiceBox.TTS.Hojo.Internal;

/// <summary>Per-request sampling state and reusable nucleus buffers; no global RNG or token-loop allocations.</summary>
internal sealed class HojoSampler(HojoRequest request, int vocabularySize = 17685)
{
    private readonly Random _random = request.Seed is { } seed ? new(seed) : new(Random.Shared.Next());
    private readonly bool[] _seen = new bool[vocabularySize];
    private readonly double[] _probabilities = new double[vocabularySize];
    private readonly int[] _order = new int[vocabularySize];

    public int Next(ReadOnlySpan<float> logits)
    {
        if (logits.Length != _seen.Length) throw new InvalidDataException("Unexpected LM vocabulary size.");
        double max = double.NegativeInfinity;
        int best = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double value = logits[i];
            if (!double.IsFinite(value)) throw new InvalidDataException("LM returned non-finite logits.");
            if (_seen[i]) value = value > 0 ? value / request.Penalty : value * request.Penalty;
            _probabilities[i] = value;
            if (value > max) { max = value; best = i; }
        }
        if (!request.Sample || request.Temperature == 0) { _seen[best] = true; return best; }
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            _probabilities[i] = Math.Exp((_probabilities[i] - max) / request.Temperature);
            sum += _probabilities[i];
            _order[i] = i;
        }
        // Sorting paired arrays avoids a comparer callback on every comparison.
        Array.Sort(_probabilities, _order);
        int first = request.TopK > 0 ? Math.Max(0, logits.Length - request.TopK) : 0;
        double topSum = sum;
        if (first > 0)
        {
            topSum = 0;
            for (int i = first; i < logits.Length; i++) topSum += _probabilities[i];
        }
        double kept = 0;
        int last = logits.Length - 1;
        for (; last >= first; last--)
        {
            kept += _probabilities[last];
            if (kept >= request.TopP * topSum) break;
        }
        first = Math.Max(first, last);
        double draw = _random.NextDouble() * kept;
        int chosen = _order[^1];
        for (int i = first; i < logits.Length; i++)
        {
            draw -= _probabilities[i];
            if (draw < 0) { chosen = _order[i]; break; }
        }
        _seen[chosen] = true;
        return chosen;
    }
}
