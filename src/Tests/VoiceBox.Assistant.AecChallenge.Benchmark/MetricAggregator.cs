namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class MetricAggregator
{
    private const double ConfidenceZ95 = 1.959963984540054;
    private const double PoissonUpperTailAlpha = 0.05;
    private const int ClusterBootstrapIterations = 4000;

    public static MetricSlice Aggregate(string name, IEnumerable<CaseResult> source)
    {
        CaseResult[] cases = source.ToArray();
        CaseResult[] valid = cases.Where(c => c.Error is null).ToArray();
        CaseResult[] positives = valid
            .Where(c => c.ExpectedNearEnd && (!c.Oracle.Available || c.Oracle.Eligible))
            .ToArray();
        CaseResult[] negatives = valid.Where(c => !c.ExpectedNearEnd).ToArray();
        int truePositive = positives.Count(c => c.TruePositive);
        int falseNegative = positives.Count(c => c.FalseNegative);
        int falsePositive = negatives.Count(c => c.FalsePositive);
        int trueNegative = negatives.Length - falsePositive;
        int early = valid.Count(c => c.EarlyFalseTrigger);
        int ungradeable = valid.Length - positives.Length - negatives.Length;
        double negativeHours = negatives.Sum(c => c.EvaluatedDurationMs) / 3_600_000.0;
        CaseResult[] oraclePositives = positives.Where(c => c.Oracle.Available && c.Oracle.Eligible).ToArray();
        double[] latencies = positives
            .Where(c => c.TruePositive && c.EligibleLatencyMs.HasValue)
            .Select(c => c.EligibleLatencyMs!.Value)
            .OrderBy(x => x)
            .ToArray();
        int precisionDenominator = truePositive + falsePositive + early;
        (double? recallLower, double? recallUpper) = ClusteredRatio95(
            positives,
            c => c.TruePositive ? (1, 1) : (0, 1),
            name,
            Wilson95(truePositive, positives.Length));
        (double? precisionLower, double? precisionUpper) = ClusteredRatio95(
            valid,
            c => c.TruePositive
                ? (1, 1)
                : c.FalsePositive || c.EarlyFalseTrigger
                    ? (0, 1)
                    : (0, 0),
            name + "/precision",
            Wilson95(truePositive, precisionDenominator));
        (double? specificityLower, double? specificityUpper) = ClusteredRatio95(
            negatives,
            c => c.FalsePositive ? (0, 1) : (1, 1),
            name + "/specificity",
            Wilson95(trueNegative, negatives.Length));
        double? poissonUpper = PoissonRateUpper95(falsePositive, negativeHours);
        double? clusteredRateUpper = ClusteredRateUpper95(negatives, name + "/rate");
        double? conservativeRateUpper =
            poissonUpper.HasValue && clusteredRateUpper.HasValue
                ? Math.Max(poissonUpper.Value, clusteredRateUpper.Value)
                : poissonUpper ?? clusteredRateUpper;

        return new MetricSlice
        {
            Name = name,
            Total = cases.Length,
            Positive = positives.Length,
            Negative = negatives.Length,
            Ungradeable = ungradeable,
            TruePositive = truePositive,
            FalseNegative = falseNegative,
            FalsePositive = falsePositive,
            TrueNegative = trueNegative,
            EarlyFalseTrigger = early,
            ErrorCount = cases.Count(c => c.Error is not null),
            ClusterCount = valid.Select(c => c.ClusterId).Distinct(StringComparer.Ordinal).Count(),
            Recall = Ratio(truePositive, truePositive + falseNegative),
            RecallLower95 = recallLower,
            RecallUpper95 = recallUpper,
            Precision = Ratio(truePositive, precisionDenominator),
            PrecisionLower95 = precisionLower,
            PrecisionUpper95 = precisionUpper,
            Specificity = Ratio(trueNegative, trueNegative + falsePositive),
            SpecificityLower95 = specificityLower,
            SpecificityUpper95 = specificityUpper,
            NegativeExposureHours = negativeHours,
            FalseTriggersPerHour = negativeHours > 0 ? falsePositive / negativeHours : null,
            FalseTriggersPerHourUpper95 = conservativeRateUpper,
            OraclePositive = oraclePositives.Length,
            LatencyCount = latencies.Length,
            LatencyCensored = oraclePositives.Length - latencies.Length,
            LatencyP50Ms = CensoredPercentile(latencies, oraclePositives.Length, 0.50),
            LatencyP95Ms = CensoredPercentile(latencies, oraclePositives.Length, 0.95),
        };
    }

    public static IReadOnlyList<MetricSlice> BuildSlices(IReadOnlyList<CaseResult> cases)
    {
        var slices = new List<MetricSlice>();
        slices.AddRange(cases
            .GroupBy(c => c.Corpus)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => Aggregate($"corpus:{g.Key}", g)));
        slices.AddRange(cases
            .GroupBy(c => $"{c.Corpus}/{c.Scenario}")
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => Aggregate($"scenario:{g.Key}", g)));
        slices.AddRange(cases
            .Where(c => c.DeviceClass is not null)
            .GroupBy(c => c.DeviceClass!)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => Aggregate($"device-class:{g.Key}", g)));
        slices.AddRange(cases
            .Where(c => c.DeviceClass is not null)
            .GroupBy(c => $"{c.DeviceClass}/{c.Scenario}")
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => Aggregate($"device-scenario:{g.Key}", g)));
        slices.AddRange(cases
            .GroupBy(c => c.CallbackSamples)
            .OrderBy(g => g.Key)
            .Select(g => Aggregate($"callback:{g.Key}-samples", g)));
        slices.AddRange(cases
            .Where(c => c.SignalToEchoRatioDb.HasValue)
            .GroupBy(c => SerBucket(c.SignalToEchoRatioDb!.Value))
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => Aggregate($"ser:{g.Key}", g)));
        AddBooleanSlices(slices, cases, "farend-nonlinear", c => c.FarEndNonlinear);
        AddBooleanSlices(slices, cases, "farend-noisy", c => c.FarEndNoisy);
        AddBooleanSlices(slices, cases, "nearend-noisy", c => c.NearEndNoisy);
        return slices;
    }

    public static IReadOnlyList<QualityGate> EvaluateGates(
        IReadOnlyList<CaseResult> cases,
        EvaluationOptions options)
    {
        var gates = new List<QualityGate>
        {
            EvaluationErrorGate(cases),
        };
        foreach (IGrouping<int, CaseResult> callbackGroup in cases
                     .GroupBy(c => c.CallbackSamples)
                     .OrderBy(g => g.Key))
        {
            string callback = $"callback:{callbackGroup.Key}";
            MetricSlice callbackMetrics = Aggregate(callback, callbackGroup);
            gates.Add(RecallGate(
                $"{callback}/acoustic-decision-recall",
                callbackMetrics,
                options));

            foreach (IGrouping<string, CaseResult> stratum in callbackGroup
                         .Where(IsGradeablePositive)
                         .GroupBy(c => $"{c.Corpus}/{c.Scenario}")
                         .OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                MetricSlice stratumMetrics = Aggregate(stratum.Key, stratum);
                gates.Add(RecallGate(
                    $"{callback}/recall:{stratum.Key}",
                    stratumMetrics,
                    options));
            }

            foreach (IGrouping<string, CaseResult> serGroup in callbackGroup
                         .Where(c => IsGradeablePositive(c) && c.SignalToEchoRatioDb.HasValue)
                         .GroupBy(c => SerBucket(c.SignalToEchoRatioDb!.Value))
                         .OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                MetricSlice serMetrics = Aggregate(serGroup.Key, serGroup);
                gates.Add(RecallGate(
                    $"{callback}/recall:ser:{serGroup.Key}",
                    serMetrics,
                    options));
            }

            gates.Add(FalseTriggerRateGate(
                $"{callback}/farend-false-triggers-per-hour",
                callbackMetrics,
                options));
            gates.Add(LatencyGate(
                $"{callback}/eligible-onset-to-decision-p95-ms",
                callbackMetrics,
                options));

            if (options.Suite == EvaluationSuite.DeviceSample)
                AddDeviceGates(gates, callback, callbackGroup, options);
        }
        return gates;
    }

    private static QualityGate EvaluationErrorGate(IReadOnlyList<CaseResult> cases)
    {
        int errorCount = cases.Count(c => c.Error is not null);
        return new QualityGate(
            "evaluation/errors",
            errorCount == 0 ? GateStatus.Passed : GateStatus.Failed,
            errorCount,
            0,
            "<=",
            errorCount == 0
                ? "All evaluation cases completed without errors."
                : $"{errorCount} evaluation case(s) failed; incomplete cases cannot satisfy enforced gates.");
    }

    private static void AddDeviceGates(
        ICollection<QualityGate> gates,
        string callback,
        IEnumerable<CaseResult> callbackCases,
        EvaluationOptions options)
    {
        CaseResult[] cases = callbackCases.ToArray();
        foreach (string deviceClass in DatasetCatalog.ExpectedDeviceClasses)
        {
            CaseResult[] deviceCases = cases
                .Where(c => string.Equals(c.DeviceClass, deviceClass, StringComparison.Ordinal))
                .ToArray();
            MetricSlice metrics = Aggregate(deviceClass, deviceCases);
            double coverage = Math.Min(
                metrics.Positive / (double)options.MinimumGatePositiveCount,
                metrics.Negative / (double)options.MinimumGateNegativeCount);
            bool sufficient =
                metrics.Positive >= options.MinimumGatePositiveCount &&
                metrics.Negative >= options.MinimumGateNegativeCount;
            gates.Add(new QualityGate(
                $"{callback}/device-coverage:{deviceClass}",
                sufficient ? GateStatus.Passed : GateStatus.Inconclusive,
                coverage,
                1,
                ">=",
                $"Need positive>={options.MinimumGatePositiveCount} and " +
                $"negative>={options.MinimumGateNegativeCount}; " +
                $"got positive={metrics.Positive}, negative={metrics.Negative}."));

            if (deviceCases.Length == 0)
                continue;

            gates.Add(RecallGate(
                $"{callback}/device:{deviceClass}/recall",
                metrics,
                options));
            gates.Add(FalseTriggerRateGate(
                $"{callback}/device:{deviceClass}/farend-false-triggers-per-hour",
                metrics,
                options));
        }
    }

    private static QualityGate RecallGate(
        string name,
        MetricSlice metrics,
        EvaluationOptions options)
    {
        if (!metrics.Recall.HasValue || metrics.Positive < options.MinimumGatePositiveCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                metrics.RecallLower95,
                options.MinimumRecall,
                ">=",
                $"Need at least {options.MinimumGatePositiveCount} positive cases; got {metrics.Positive}.");
        }

        if (metrics.Recall < options.MinimumRecall)
        {
            return new QualityGate(
                name,
                GateStatus.Failed,
                metrics.RecallLower95,
                options.MinimumRecall,
                ">=",
                $"Point recall={metrics.Recall:P2}, 95% clustered bootstrap CI=" +
                $"{Interval(metrics.RecallLower95, metrics.RecallUpper95)}, " +
                $"n={metrics.Positive}, clusters={metrics.ClusterCount}.");
        }

        GateStatus status = metrics.RecallLower95 >= options.MinimumRecall
            ? GateStatus.Passed
            : GateStatus.Inconclusive;
        return new QualityGate(
            name,
            status,
            metrics.RecallLower95,
            options.MinimumRecall,
            ">=",
            $"Point recall={metrics.Recall:P2}, 95% clustered bootstrap CI=" +
            $"{Interval(metrics.RecallLower95, metrics.RecallUpper95)}, " +
            $"n={metrics.Positive}, clusters={metrics.ClusterCount}.");
    }

    private static QualityGate FalseTriggerRateGate(
        string name,
        MetricSlice metrics,
        EvaluationOptions options)
    {
        if (!metrics.FalseTriggersPerHour.HasValue ||
            !metrics.FalseTriggersPerHourUpper95.HasValue ||
            metrics.Negative < options.MinimumGateNegativeCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                metrics.FalseTriggersPerHourUpper95,
                options.MaximumFalseTriggersPerHour,
                "<=",
                $"Need at least {options.MinimumGateNegativeCount} negative cases; got {metrics.Negative}.");
        }

        if (metrics.FalseTriggersPerHour > options.MaximumFalseTriggersPerHour)
        {
            return new QualityGate(
                name,
                GateStatus.Failed,
                metrics.FalseTriggersPerHourUpper95,
                options.MaximumFalseTriggersPerHour,
                "<=",
                $"Point rate={metrics.FalseTriggersPerHour:F3}/h, one-sided 95% Poisson upper=" +
                $"{metrics.FalseTriggersPerHourUpper95:F3}/h, exposure={metrics.NegativeExposureHours:F3}h.");
        }

        GateStatus status =
            metrics.FalseTriggersPerHourUpper95 <= options.MaximumFalseTriggersPerHour
                ? GateStatus.Passed
                : GateStatus.Inconclusive;
        return new QualityGate(
            name,
            status,
            metrics.FalseTriggersPerHourUpper95,
            options.MaximumFalseTriggersPerHour,
            "<=",
            $"Point rate={metrics.FalseTriggersPerHour:F3}/h, one-sided 95% Poisson upper=" +
            $"{metrics.FalseTriggersPerHourUpper95:F3}/h, exposure={metrics.NegativeExposureHours:F3}h.");
    }

    private static QualityGate LatencyGate(
        string name,
        MetricSlice metrics,
        EvaluationOptions options)
    {
        if (metrics.OraclePositive < options.MinimumGateLatencyCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                metrics.LatencyP95Ms,
                options.MaximumP95EligibleLatencyMs,
                "<=",
                $"Need at least {options.MinimumGateLatencyCount} oracle-positive cases; " +
                $"got {metrics.OraclePositive}.");
        }

        if (!metrics.LatencyP95Ms.HasValue)
        {
            return new QualityGate(
                name,
                GateStatus.Failed,
                null,
                options.MaximumP95EligibleLatencyMs,
                "<=",
                $"P95 is right-censored by missed/early detections; observed={metrics.LatencyCount}, " +
                $"censored={metrics.LatencyCensored}, oracle-positive={metrics.OraclePositive}.");
        }

        return new QualityGate(
            name,
            metrics.LatencyP95Ms <= options.MaximumP95EligibleLatencyMs
                ? GateStatus.Passed
                : GateStatus.Failed,
            metrics.LatencyP95Ms,
            options.MaximumP95EligibleLatencyMs,
            "<=",
            $"Right-censored empirical percentile; observed={metrics.LatencyCount}, " +
            $"censored={metrics.LatencyCensored}, oracle-positive={metrics.OraclePositive}.");
    }

    private static double? Ratio(int numerator, int denominator)
        => denominator > 0 ? numerator / (double)denominator : null;

    private static bool IsGradeablePositive(CaseResult result)
        => result.Error is null &&
           result.ExpectedNearEnd &&
           (!result.Oracle.Available || result.Oracle.Eligible);

    private static void AddBooleanSlices(
        ICollection<MetricSlice> destination,
        IEnumerable<CaseResult> cases,
        string name,
        Func<CaseResult, bool?> selector)
    {
        foreach (IGrouping<bool, CaseResult> group in cases
                     .Where(c => selector(c).HasValue)
                     .GroupBy(c => selector(c)!.Value)
                     .OrderBy(g => g.Key))
        {
            destination.Add(Aggregate($"{name}:{group.Key.ToString().ToLowerInvariant()}", group));
        }
    }

    private static string SerBucket(double serDb)
        => serDb switch
        {
            <= -6 => "[-10,-6]dB",
            <= -1 => "[-5,-1]dB",
            <= 4 => "[0,4]dB",
            _ => "[5,10]dB",
        };

    private static (double? Lower, double? Upper) Wilson95(int successes, int total)
    {
        if (total <= 0)
            return (null, null);

        double n = total;
        double p = successes / n;
        double z2 = ConfidenceZ95 * ConfidenceZ95;
        double denominator = 1 + z2 / n;
        double center = (p + z2 / (2 * n)) / denominator;
        double margin =
            ConfidenceZ95 *
            Math.Sqrt((p * (1 - p) + z2 / (4 * n)) / n) /
            denominator;
        return (Math.Max(0, center - margin), Math.Min(1, center + margin));
    }

    private static (double? Lower, double? Upper) ClusteredRatio95(
        IReadOnlyList<CaseResult> cases,
        Func<CaseResult, (int Success, int Total)> selector,
        string seedKey,
        (double? Lower, double? Upper) fallback)
    {
        IGrouping<string, CaseResult>[] clusters = cases
            .GroupBy(c => c.ClusterId, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToArray();
        if (clusters.Length < 5)
            return fallback;

        var random = new Random(StableSeed(seedKey));
        var estimates = new List<double>(ClusterBootstrapIterations);
        for (int iteration = 0; iteration < ClusterBootstrapIterations; iteration++)
        {
            int success = 0;
            int total = 0;
            for (int draw = 0; draw < clusters.Length; draw++)
            {
                foreach (CaseResult item in clusters[random.Next(clusters.Length)])
                {
                    (int itemSuccess, int itemTotal) = selector(item);
                    success += itemSuccess;
                    total += itemTotal;
                }
            }
            if (total > 0)
                estimates.Add(success / (double)total);
        }
        if (estimates.Count < ClusterBootstrapIterations / 2)
            return fallback;

        estimates.Sort();
        double bootstrapLower = EmpiricalPercentile(estimates, 0.025);
        double bootstrapUpper = EmpiricalPercentile(estimates, 0.975);
        // A perfect cluster outcome produces a degenerate bootstrap distribution.
        // Retain the Wilson lower/upper bound as a finite-sample guard in that case.
        return (
            fallback.Lower.HasValue
                ? Math.Min(fallback.Lower.Value, bootstrapLower)
                : bootstrapLower,
            fallback.Upper.HasValue
                ? Math.Max(fallback.Upper.Value, bootstrapUpper)
                : bootstrapUpper);
    }

    private static double? ClusteredRateUpper95(
        IReadOnlyList<CaseResult> negatives,
        string seedKey)
    {
        IGrouping<string, CaseResult>[] clusters = negatives
            .GroupBy(c => c.ClusterId, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToArray();
        if (clusters.Length < 2)
            return null;

        var random = new Random(StableSeed(seedKey));
        var estimates = new List<double>(ClusterBootstrapIterations);
        for (int iteration = 0; iteration < ClusterBootstrapIterations; iteration++)
        {
            int events = 0;
            double hours = 0;
            for (int draw = 0; draw < clusters.Length; draw++)
            {
                foreach (CaseResult item in clusters[random.Next(clusters.Length)])
                {
                    if (item.FalsePositive)
                        events++;
                    hours += item.EvaluatedDurationMs / 3_600_000.0;
                }
            }
            if (hours > 0)
                estimates.Add(events / hours);
        }
        estimates.Sort();
        return estimates.Count == 0
            ? null
            : EmpiricalPercentile(estimates, 0.95);
    }

    private static double EmpiricalPercentile(IReadOnlyList<double> sorted, double percentile)
    {
        double position = percentile * (sorted.Count - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        if (lower == upper)
            return sorted[lower];
        double fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    private static int StableSeed(string value)
    {
        unchecked
        {
            int hash = 17;
            foreach (char c in value)
                hash = hash * 31 + c;
            return hash;
        }
    }

    private static double? PoissonRateUpper95(int events, double exposureHours)
    {
        if (events < 0 || !double.IsFinite(exposureHours) || exposureHours <= 0)
            return null;

        double lower = 0;
        double upper = Math.Max(1, events + 1);
        while (LogPoissonCdf(events, upper) > Math.Log(PoissonUpperTailAlpha))
            upper *= 2;

        for (int i = 0; i < 80; i++)
        {
            double midpoint = (lower + upper) / 2;
            if (LogPoissonCdf(events, midpoint) > Math.Log(PoissonUpperTailAlpha))
                lower = midpoint;
            else
                upper = midpoint;
        }
        return upper / exposureHours;
    }

    private static double LogPoissonCdf(int events, double mean)
    {
        if (mean <= 0)
            return 0;

        double logTerm = -mean;
        double logSum = logTerm;
        double logMean = Math.Log(mean);
        for (int i = 1; i <= events; i++)
        {
            logTerm += logMean - Math.Log(i);
            logSum = LogAdd(logSum, logTerm);
        }
        return logSum;
    }

    private static double LogAdd(double left, double right)
    {
        double maximum = Math.Max(left, right);
        return maximum + Math.Log(Math.Exp(left - maximum) + Math.Exp(right - maximum));
    }

    private static double? CensoredPercentile(
        double[] sortedObserved,
        int totalIncludingCensored,
        double percentile)
    {
        if (sortedObserved.Length == 0 || totalIncludingCensored <= 0)
            return null;

        double position = percentile * (totalIncludingCensored - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        if (upper >= sortedObserved.Length)
            return null;
        if (lower == upper)
            return sortedObserved[lower];
        double fraction = position - lower;
        return sortedObserved[lower] + (sortedObserved[upper] - sortedObserved[lower]) * fraction;
    }

    private static string Interval(double? lower, double? upper)
        => lower.HasValue && upper.HasValue
            ? $"[{lower.Value:P2}, {upper.Value:P2}]"
            : "n/a";
}
