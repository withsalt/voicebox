using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Estimation;

/// <summary>
/// Stream-local, render-conditioned echo statistics. The profile contains no device
/// identity or fixed transport delay and retains only bounded decaying histograms.
/// </summary>
internal sealed class RuntimeEchoProfile
{
    private const int RenderBucketCount = 6;
    private const double MinimumRenderDb = -72;
    private const double MaximumRenderDb = 0;
    private const double MinimumRenderRms = 1e-5;
    private const double HalfLifeMs = 120_000;
    private const double ReadyTrustedEchoMs = 320;
    private const double ReadyRelevantEchoMs = 160;
    private const double UpperQuantile = 0.995;

    private readonly DecayingHistogram[] _captureDbByRender = CreateHistograms(180, -100, 0);
    private readonly DecayingHistogram[] _postAecDbByRender = CreateHistograms(180, -100, 0);
    private readonly DecayingHistogram[] _residualRatioDbByRender = CreateHistograms(160, -60, 20);
    private readonly DecayingHistogram _captureDbPrior = new(180, -100, 0);
    private readonly DecayingHistogram _postAecDbPrior = new(180, -100, 0);
    private readonly DecayingHistogram _residualRatioDbPrior = new(160, -60, 20);
    private readonly DecayingHistogram _probability = new(100, 0, 1);
    private readonly DecayingHistogram _smoothedProbability = new(100, 0, 1);

    private double _trustedEchoDurationMs;

    public void Advance(double elapsedMs)
    {
        if (!double.IsFinite(elapsedMs) || elapsedMs <= 0)
            return;

        double retention = Math.Pow(0.5, elapsedMs / HalfLifeMs);
        _trustedEchoDurationMs *= retention;
        _captureDbPrior.Decay(retention);
        _postAecDbPrior.Decay(retention);
        _residualRatioDbPrior.Decay(retention);
        _probability.Decay(retention);
        _smoothedProbability.Decay(retention);
        for (int i = 0; i < RenderBucketCount; i++)
        {
            _captureDbByRender[i].Decay(retention);
            _postAecDbByRender[i].Decay(retention);
            _residualRatioDbByRender[i].Decay(retention);
        }
    }

    public void ObserveTrustedEcho(
        double renderRms,
        double captureRms,
        double postAecRms,
        double residualRatio,
        float probability,
        float smoothedProbability,
        double frameDurationMs)
    {
        if (!double.IsFinite(renderRms) || renderRms <= MinimumRenderRms ||
            !double.IsFinite(captureRms) || captureRms <= 0 ||
            !double.IsFinite(postAecRms) || postAecRms < 0 ||
            !double.IsFinite(residualRatio) || residualRatio < 0 ||
            !float.IsFinite(probability) || !float.IsFinite(smoothedProbability) ||
            !double.IsFinite(frameDurationMs) || frameDurationMs <= 0)
        {
            return;
        }

        double weight = Math.Clamp(frameDurationMs, 1, 100);
        double position = RenderBucketPosition(RmsToDb(renderRms));
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(lower + 1, RenderBucketCount - 1);
        double upperWeight = position - lower;
        double lowerWeight = 1 - upperWeight;
        double captureDb = RmsToDb(captureRms);
        double postAecDb = RmsToDb(postAecRms);
        double residualRatioDb = RmsToDb(residualRatio);

        _trustedEchoDurationMs += weight;
        AddInterpolated(_captureDbByRender, lower, upper, captureDb, weight, lowerWeight, upperWeight);
        AddInterpolated(_postAecDbByRender, lower, upper, postAecDb, weight, lowerWeight, upperWeight);
        AddInterpolated(_residualRatioDbByRender, lower, upper, residualRatioDb, weight, lowerWeight, upperWeight);
        _captureDbPrior.Add(captureDb, weight * 0.20);
        _postAecDbPrior.Add(postAecDb, weight * 0.20);
        _residualRatioDbPrior.Add(residualRatioDb, weight * 0.20);
        _probability.Add(Math.Clamp(probability, 0, 1), weight);
        _smoothedProbability.Add(Math.Clamp(smoothedProbability, 0, 1), weight);
    }

    public RuntimeEchoProfileSnapshot Snapshot(double renderRms)
    {
        if (_trustedEchoDurationMs <= 0)
            return default;

        double position = RenderBucketPosition(RmsToDb(Math.Max(renderRms, MinimumRenderRms)));
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(lower + 1, RenderBucketCount - 1);
        double upperWeight = position - lower;
        double lowerWeight = 1 - upperWeight;
        double relevantDurationMs =
            _captureDbByRender[lower].EffectiveWeight * lowerWeight +
            _captureDbByRender[upper].EffectiveWeight * upperWeight;
        double reliability = Math.Min(
            Math.Clamp(_trustedEchoDurationMs / ReadyTrustedEchoMs, 0, 1),
            Math.Clamp(relevantDurationMs / ReadyRelevantEchoMs, 0, 1));

        double captureDb = InterpolatedQuantile(
            _captureDbByRender,
            _captureDbPrior,
            lower,
            upper,
            lowerWeight,
            upperWeight);
        double postAecDb = InterpolatedQuantile(
            _postAecDbByRender,
            _postAecDbPrior,
            lower,
            upper,
            lowerWeight,
            upperWeight);
        double residualRatioDb = InterpolatedQuantile(
            _residualRatioDbByRender,
            _residualRatioDbPrior,
            lower,
            upper,
            lowerWeight,
            upperWeight);

        return new RuntimeEchoProfileSnapshot(
            reliability,
            _trustedEchoDurationMs,
            relevantDurationMs,
            DbToRatio(captureDb),
            DbToRatio(postAecDb),
            DbToRatio(residualRatioDb),
            (float)Math.Clamp(_probability.Quantile(UpperQuantile), 0, 1),
            (float)Math.Clamp(_smoothedProbability.Quantile(UpperQuantile), 0, 1));
    }

    public void Reset()
    {
        _trustedEchoDurationMs = 0;
        _captureDbPrior.Reset();
        _postAecDbPrior.Reset();
        _residualRatioDbPrior.Reset();
        _probability.Reset();
        _smoothedProbability.Reset();
        for (int i = 0; i < RenderBucketCount; i++)
        {
            _captureDbByRender[i].Reset();
            _postAecDbByRender[i].Reset();
            _residualRatioDbByRender[i].Reset();
        }
    }

    private static DecayingHistogram[] CreateHistograms(int bins, double minimum, double maximum)
    {
        var histograms = new DecayingHistogram[RenderBucketCount];
        for (int i = 0; i < histograms.Length; i++)
            histograms[i] = new DecayingHistogram(bins, minimum, maximum);
        return histograms;
    }

    private static void AddInterpolated(
        DecayingHistogram[] histograms,
        int lower,
        int upper,
        double value,
        double weight,
        double lowerWeight,
        double upperWeight)
    {
        histograms[lower].Add(value, weight * lowerWeight);
        if (upper != lower)
            histograms[upper].Add(value, weight * upperWeight);
    }

    private static double InterpolatedQuantile(
        DecayingHistogram[] histograms,
        DecayingHistogram prior,
        int lower,
        int upper,
        double lowerWeight,
        double upperWeight)
    {
        double priorValue = prior.Quantile(UpperQuantile);
        double lowerValue = histograms[lower].EffectiveWeight > 0
            ? histograms[lower].Quantile(UpperQuantile)
            : priorValue;
        double upperValue = histograms[upper].EffectiveWeight > 0
            ? histograms[upper].Quantile(UpperQuantile)
            : priorValue;
        return lowerValue * lowerWeight + upperValue * upperWeight;
    }

    private static double RenderBucketPosition(double renderDb)
        => Math.Clamp(
            (renderDb - MinimumRenderDb) * (RenderBucketCount - 1) /
            (MaximumRenderDb - MinimumRenderDb),
            0,
            RenderBucketCount - 1);

    private static double RmsToDb(double rms)
        => 20 * Math.Log10(Math.Max(rms, 1e-8));

    private static double DbToRatio(double value)
        => Math.Pow(10, value / 20);

    /// <summary>
    /// Weighted histogram with deferred exponential decay. A shared scale factor avoids updating
    /// every bin on each frame; it cancels out when selecting a quantile.
    /// </summary>
    private sealed class DecayingHistogram
    {
        /// <summary>
        /// Lower bound for the carried scale before the deferred decay is materialized
        /// into the bins. At a 120 s half-life and a 20 ms frame the factor needs roughly
        /// 13 hours of uninterrupted playback to reach this, so rebasing is rare.
        /// </summary>
        private const double MinimumCarriedScale = 1e-12;

        private readonly double[] _weights;
        private readonly double _minimum;
        private readonly double _maximum;
        private readonly double _scale;
        private double _carriedScale = 1;
        private double _carriedEffectiveWeight;

        public DecayingHistogram(int bins, double minimum, double maximum)
        {
            if (bins < 2)
                throw new ArgumentOutOfRangeException(nameof(bins));
            if (!double.IsFinite(minimum) || !double.IsFinite(maximum) || maximum <= minimum)
                throw new ArgumentOutOfRangeException(nameof(maximum));

            _weights = new double[bins];
            _minimum = minimum;
            _maximum = maximum;
            _scale = (bins - 1) / (maximum - minimum);
        }

        public double EffectiveWeight => _carriedEffectiveWeight * _carriedScale;

        public void Add(double value, double weight)
        {
            if (!double.IsFinite(value) || !double.IsFinite(weight) || weight <= 0)
                return;

            int index = (int)Math.Round((Math.Clamp(value, _minimum, _maximum) - _minimum) * _scale);
            double carried = weight / _carriedScale;
            _weights[Math.Clamp(index, 0, _weights.Length - 1)] += carried;
            _carriedEffectiveWeight += carried;
        }

        public void Decay(double retention)
        {
            if (!double.IsFinite(retention) || retention is < 0 or > 1)
                return;

            // A zero retention is a full reset; rebasing immediately keeps the carried
            // scale strictly positive so Add never divides by zero.
            if (retention == 0)
            {
                Reset();
                return;
            }

            _carriedScale *= retention;
            if (_carriedScale < MinimumCarriedScale)
                Rebase();
        }

        /// <summary>Folds the carried scale back into the bins and restores it to one.</summary>
        private void Rebase()
        {
            double carriedScale = _carriedScale;
            for (int i = 0; i < _weights.Length; i++)
                _weights[i] *= carriedScale;
            _carriedEffectiveWeight *= carriedScale;
            _carriedScale = 1;
        }

        public double Quantile(double quantile)
        {
            // Both the target and the running sum stay in the carried domain, where the
            // carried scale cancels. That keeps the selected bin identical to eager decay
            // and avoids scaling every bin on the way.
            if (_carriedEffectiveWeight <= 0)
                return _minimum;

            double target = Math.Clamp(quantile, 0, 1) * _carriedEffectiveWeight;
            double cumulative = 0;
            for (int i = 0; i < _weights.Length; i++)
            {
                cumulative += _weights[i];
                if (cumulative >= target)
                    return _minimum + i / _scale;
            }
            return _maximum;
        }

        public void Reset()
        {
            Array.Clear(_weights);
            _carriedEffectiveWeight = 0;
            _carriedScale = 1;
        }
    }
}
