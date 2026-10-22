using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class ChineseTextMetrics
{
    public static (int Errors, int ReferenceCharacters, double? Cer, bool Exact) Compare(
        string reference,
        string hypothesis)
    {
        string normalizedReference = Normalize(reference);
        string normalizedHypothesis = Normalize(hypothesis);
        int errors = Levenshtein(normalizedReference, normalizedHypothesis);
        return (
            errors,
            normalizedReference.Length,
            normalizedReference.Length > 0
                ? errors / (double)normalizedReference.Length
                : null,
            string.Equals(
                normalizedReference,
                normalizedHypothesis,
                StringComparison.Ordinal));
    }

    public static string Normalize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;
        string compatibility = text.Normalize(NormalizationForm.FormKC);
        var builder = new StringBuilder(compatibility.Length);
        foreach (char c in compatibility)
        {
            if (char.IsWhiteSpace(c) ||
                char.IsPunctuation(c) ||
                char.IsSymbol(c) ||
                char.IsControl(c))
            {
                continue;
            }
            builder.Append(char.ToLowerInvariant(c));
        }
        return builder.ToString();
    }

    private static int Levenshtein(string left, string right)
    {
        if (left.Length == 0)
            return right.Length;
        if (right.Length == 0)
            return left.Length;

        var previous = new int[right.Length + 1];
        var current = new int[right.Length + 1];
        for (int j = 0; j <= right.Length; j++)
            previous[j] = j;
        for (int i = 1; i <= left.Length; i++)
        {
            current[0] = i;
            for (int j = 1; j <= right.Length; j++)
            {
                int substitution = left[i - 1] == right[j - 1] ? 0 : 1;
                current[j] = Math.Min(
                    Math.Min(current[j - 1] + 1, previous[j] + 1),
                    previous[j - 1] + substitution);
            }
            (previous, current) = (current, previous);
        }
        return previous[right.Length];
    }
}

internal static class EndToEndMetrics
{
    public static EndToEndSummary Aggregate(IEnumerable<EndToEndCaseResult> source)
    {
        EndToEndCaseResult[] cases = source.ToArray();
        ClusterObservation[] clusters = cases
            .GroupBy(c => c.ClusterId, StringComparer.Ordinal)
            .Select(group => new ClusterObservation(group.Key, group.ToArray()))
            .ToArray();
        ClusterObservation[] valid = clusters.Where(c => !c.HasError).ToArray();
        ClusterObservation[] positive = valid.Where(c => c.ExpectHeard).ToArray();
        ClusterObservation[] negative = valid.Where(c => !c.ExpectHeard).ToArray();
        ClusterObservation[] acceptedExpected = valid
            .Where(c => c.ExpectAccepted)
            .ToArray();
        ClusterObservation[] duckExpected = valid.Where(c => c.ExpectDuck).ToArray();
        ClusterObservation[] stopExpected = valid
            .Where(c => c.ExpectPlaybackStop)
            .ToArray();
        ClusterObservation[] resumeExpected = valid
            .Where(c => c.ExpectPlaybackResume)
            .ToArray();
        // Latency percentiles are taken over the clusters that produced an observation; the censored
        // counts alongside them record how many did not. Folding misses into the percentile as well
        // charges one failure twice and, past a handful of misses, collapses the percentile to "no
        // value" - which reports nothing about latency at all while the recall metric has already
        // reported the misses.
        double? heardP50 = ObservedPercentile(
            positive,
            observation => observation.MaxLatency(c => c.HeardLatencyMs),
            0.50);
        double? heardP95 = ObservedPercentile(
            positive,
            observation => observation.MaxLatency(c => c.HeardLatencyMs),
            0.95);
        double? stopP95 = ObservedPercentile(
            stopExpected,
            observation => observation.MaxLatency(c => c.StopLatencyMs),
            0.95);
        double? audibleStopP95 = ObservedPercentile(
            stopExpected,
            observation => observation.MaxLatency(c => c.AudibleStopLatencyMs),
            0.95);
        double? audibleTailP95 = ObservedPercentile(
            stopExpected,
            observation => observation.MaxLatency(c => c.AudibleTailLatencyMs),
            0.95);
        int heardCensored = CensoredCount(positive, observation => observation.HeardSuccess);
        int stopCensored = CensoredCount(stopExpected, observation => observation.StopSuccess);
        int audibleStopCensored = CensoredCount(
            stopExpected,
            observation => observation.AudibleStopSuccess);
        int audibleTailCensored = CensoredCount(
            stopExpected,
            observation => observation.AudibleTailSuccess);
        EndToEndCaseResult[] acceptedCases = valid
            .SelectMany(observation => observation.Cases)
            .Where(c => c.Accepted)
            .ToArray();
        EndToEndCaseResult[] negativeCases = negative
            .SelectMany(observation => observation.Cases)
            .ToArray();
        int negativeFalseAccepts = negativeCases.Count(c => c.Accepted);
        double negativeHours = negativeCases.Sum(c => c.AudioDurationMs) / 3_600_000.0;
        double[] clusterCer = acceptedExpected
            .Select(observation => observation.Cases
                .Where(c => c.ExpectAccepted && c.ReferenceCharacters > 0)
                .ToArray())
            .Where(scored => scored.Length > 0)
            .Select(scored =>
                scored.Sum(c => c.CharacterErrors) /
                (double)scored.Sum(c => c.ReferenceCharacters))
            .ToArray();
        return new EndToEndSummary
        {
            Total = cases.Length,
            ClusterCount = clusters.Length,
            Positive = positive.Length,
            Negative = negative.Length,
            ErrorCount = cases.Count(c => c.Error is not null),
            HeardPositive = positive.Count(c => c.HeardSuccess),
            DuckedPositive = duckExpected.Count(c => c.DuckSuccess),
            AcceptedPositive = acceptedExpected.Count(c => c.AcceptedSuccess),
            StoppedPositive = stopExpected.Count(c => c.StopSuccess),
            ResumedPositive = resumeExpected.Count(c => c.ResumeSuccess),
            RejectedPositive = positive.Count(c => c.Cases.Any(item => item.Rejected)),
            EarlyFalseTrigger = positive.Count(c => c.Cases.Any(item => item.EarlyFalseTrigger)),
            EchoOnlyHeard = negative.Count(c => c.Cases.Any(item => item.Heard)),
            EchoOnlyAccepted = valid
                .Where(c => !c.ExpectAccepted)
                .Count(c => c.Cases.Any(item => item.Accepted)),
            HeardRecall = Ratio(positive.Count(c => c.HeardSuccess), positive.Length),
            DuckRecall = Ratio(duckExpected.Count(c => c.DuckSuccess), duckExpected.Length),
            AcceptedRecall = Ratio(
                acceptedExpected.Count(c => c.AcceptedSuccess),
                acceptedExpected.Length),
            PlaybackStopRecall = Ratio(
                stopExpected.Count(c => c.StopSuccess),
                stopExpected.Length),
            PlaybackResumeRecall = Ratio(
                resumeExpected.Count(c => c.ResumeSuccess),
                resumeExpected.Length),
            AggregateCharacterErrorRate = clusterCer.Length > 0
                ? clusterCer.Average()
                : null,
            ExactMatchRate = Ratio(
                acceptedExpected.Count(c => c.Cases
                    .Where(item => item.ExpectAccepted)
                    .All(item => item.ExactMatch)),
                acceptedExpected.Length),
            EmptyRecognitionRate = Ratio(
                acceptedExpected.Count(c => c.Cases
                    .Where(item => item.ExpectAccepted)
                    .Any(item => item.EmptyRecognition)),
                acceptedExpected.Length),
            HeardLatencyP50Ms = heardP50,
            HeardLatencyP95Ms = heardP95,
            StopLatencyP95Ms = stopP95,
            AudibleStopLatencyP95Ms = audibleStopP95,
            AudibleTailLatencyP95Ms = audibleTailP95,
            HeardLatencyCensored = heardCensored,
            StopLatencyCensored = stopCensored,
            AudibleStopLatencyCensored = audibleStopCensored,
            AudibleTailLatencyCensored = audibleTailCensored,
            // Latency percentiles below are taken over the clusters that actually produced an
            // observation. Misses are already counted by the recall metrics; folding them into the
            // latency percentile as well double-counts the same failure and, once enough clusters
            // miss, collapses the percentile to "censored" - which reports nothing at all about
            // latency. Reporting recall and conditional latency separately is strictly more
            // informative, and the *Censored counts stay in the report so the conditioning is visible.
            DuckLatencyP95Ms = ObservedPercentile(
                duckExpected,
                observation => observation.MaxLatency(c => c.DuckLatencyMs),
                0.95),
            AudibleDuckLatencyP95Ms = ObservedPercentile(
                duckExpected,
                observation => observation.MaxLatency(c => c.AudibleDuckLatencyMs),
                0.95),
            DrainCompleteLatencyP95Ms = ObservedPercentile(
                stopExpected,
                observation => observation.MaxLatency(c => c.DrainCompleteLatencyMs),
                0.95),
            DeviceTailLatencyP95Ms = ObservedPercentile(
                stopExpected,
                observation => observation.MaxLatency(c => c.DeviceTailLatencyMs),
                0.95),
            TextValidatedAcceptRate = Ratio(
                acceptedCases.Count(c => IsSpeakerAuthorized(c)),
                acceptedCases.Length),
            TimeoutAcceptRate = Ratio(
                acceptedCases.Count(c => IsAcousticallyAuthorized(c)),
                acceptedCases.Length),
            FalseAcceptPerHour = negativeHours > 0
                ? negativeFalseAccepts / negativeHours
                : null,
            NegativeAudioHours = negativeHours,
        };
    }

    /// <summary>
    /// With KWS enabled an accept is authorized by the candidate's own speaker Match
    /// (<c>speaker-match</c>); with KWS disabled by the acoustic Interrupt (<c>audio-confirmed</c>).
    /// Recognition text never carries the stop decision any more.
    /// </summary>
    private static bool IsSpeakerAuthorized(EndToEndCaseResult result)
        => result.Accepted &&
           string.Equals(result.ValidationReason, "speaker-match", StringComparison.Ordinal);

    private static bool IsAcousticallyAuthorized(EndToEndCaseResult result)
        => result.Accepted &&
           string.Equals(result.ValidationReason, "audio-confirmed", StringComparison.Ordinal);

    public static IReadOnlyList<EndToEndSummarySlice> BuildSlices(
        IReadOnlyList<EndToEndCaseResult> cases)
    {
        var slices = new List<EndToEndSummarySlice>();
        slices.AddRange(cases
            .GroupBy(c => c.Corpus)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"corpus:{g.Key}",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .GroupBy(c => SerBucket(c.SignalToEchoRatioDb))
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"ser:{g.Key}",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .GroupBy(c => c.CallbackSamples)
            .OrderBy(g => g.Key)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"callback:{g.Key}-samples",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .GroupBy(c => c.Scenario)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"scenario:{g.Key}",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .Where(c => !string.IsNullOrWhiteSpace(c.GeometryId))
            .GroupBy(c => c.GeometryId)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"geometry:{g.Key}",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .GroupBy(c => c.NoiseKind ?? "none")
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"noise:{g.Key}",
                Summary = Aggregate(g),
            }));
        slices.AddRange(cases
            .GroupBy(c => c.RenderGapDurationMs > 0 ? "present" : "none")
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new EndToEndSummarySlice
            {
                Name = $"render-gap:{g.Key}",
                Summary = Aggregate(g),
            }));
        return slices;
    }

    /// <summary>Product requirement for detection latency.</summary>
    private const double HeardLatencyBudgetMs = 1000;

    /// <summary>
    /// Irreducible device cost between a control decision and the loudspeaker reflecting it:
    /// the committed output latency plus the playback gain ramp. No implementation can beat it.
    /// </summary>
    private const double DeviceOutputBudgetMs = 35;

    /// <summary>
    /// Identity authorization budget. With KWS disabled the acoustic Interrupt stops playback at
    /// once; with KWS enabled the stop waits for the candidate's first scored speaker window
    /// (<c>Speaker.Confirmation.MinimumSpeechMs</c> of speech plus inference). No recognition text
    /// and no timer are on the stop path any more.
    /// </summary>
    private const double IdentityAuthorizationBudgetMs = 1000;

    public static IReadOnlyList<QualityGate> EvaluateGates(EndToEndSummary summary)
    {
        return
        [
            new QualityGate(
                "e2e/evaluation-errors",
                summary.ErrorCount == 0 ? GateStatus.Passed : GateStatus.Failed,
                summary.ErrorCount,
                0,
                "<=",
                summary.ErrorCount == 0
                    ? "All end-to-end cases completed without errors."
                    : $"{summary.ErrorCount} end-to-end case(s) failed; incomplete cases cannot satisfy enforced gates."),

            // Detection. Duck/accept/stop recall are deliberately NOT gated: they are produced by
            // the same decision as Heard and in every observed run they carry an identical value, so
            // gating them added three gates and zero information. They remain in the summary and in
            // every slice.
            RatioGate(
                "e2e/heard-recall",
                summary.HeardRecall,
                summary.Positive,
                minimumCount: 30,
                required: 0.85),

            // Latency, conditional on the event occurring. Misses are accounted for by heard-recall.
            ObservedLatencyGate(
                "e2e/heard-latency-p95-ms",
                summary.HeardLatencyP95Ms,
                summary.Positive,
                minimumCount: 30,
                required: HeardLatencyBudgetMs),

            // Hard stop is the only real-time arm now (ducking is retired): the owner-scoped stop is
            // issued at authorization, before recognition. Budget = detection + identity
            // authorization + device cost. A scheduling regression in the control loop moves it.
            ObservedLatencyGate(
                "e2e/drain-complete-latency-p95-ms",
                summary.DrainCompleteLatencyP95Ms,
                summary.Positive,
                minimumCount: 30,
                required: HeardLatencyBudgetMs + IdentityAuthorizationBudgetMs + DeviceOutputBudgetMs),

            // False accepts as a rate, so the figure does not move with dataset size. Matches the
            // acoustic suite's farend-false-triggers-per-hour target. These are adversarial hours.
            MaximumGate(
                "e2e/false-accept-per-hour",
                summary.FalseAcceptPerHour,
                required: 1,
                summary.Negative,
                minimumCount: 10),
        ];
    }

    /// <summary>
    /// Metrics that are reported and compared but not enforced, with the reason. Keeping them named
    /// here stops a known-unreachable or known-confounded gate from being silently forgotten.
    /// </summary>
    public static IReadOnlyList<string> TrackedNotEnforced =>
    [
        "e2e/accepted-recall: collinear with heard-recall in every observed run.",
        "e2e/duck-recall and e2e/audible-duck-latency-p95-ms: ducking is retired; the reply is stopped (owner-scoped) or left untouched, never attenuated.",
        "e2e/playback-stop-recall: collinear with heard-recall.",
        "e2e/audible-stop-latency-p95-ms: dominated by the dataset RIR's decay to -60 dBFS, so it measures the room rather than the product. drain-complete-latency-p95-ms gates the device-side stop.",
        "e2e/aggregate-cer: conflates recognition quality with barge-in effectiveness; attribution requires a matched no-echo reference run.",
        "e2e/speaker-authorized-accept-rate vs acoustic-authorized-accept-rate: reports which identity gate issued the stop (KWS on: speaker-match; KWS off: audio-confirmed); recognition text no longer carries the decision.",
    ];

    /// <summary>
    /// Gates a latency percentile taken over the clusters that produced an observation. Absence of
    /// any observation is inconclusive rather than failed, because the corresponding recall gate is
    /// what reports missing events.
    /// </summary>
    private static QualityGate ObservedLatencyGate(
        string name,
        double? actual,
        int count,
        int minimumCount,
        double required)
    {
        if (count < minimumCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                actual,
                required,
                "<=",
                $"Need at least {minimumCount} source clusters; got {count}.");
        }
        if (!actual.HasValue)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                null,
                required,
                "<=",
                "No observation was available; see the matching recall gate.");
        }
        return new QualityGate(
            name,
            actual <= required ? GateStatus.Passed : GateStatus.Failed,
            actual,
            required,
            "<=",
            $"Percentile over observed clusters; misses are gated by recall, not counted here.");
    }

    private static QualityGate RatioGate(
        string name,
        double? actual,
        int count,
        int minimumCount,
        double required)
    {
        if (!actual.HasValue || count < minimumCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                actual,
                required,
                ">=",
                $"Need at least {minimumCount} cases; got {count}.");
        }
        return new QualityGate(
            name,
            actual >= required ? GateStatus.Passed : GateStatus.Failed,
            actual,
            required,
            ">=",
            $"Point estimate over {count} independent source clusters.");
    }

    private static QualityGate MaximumGate(
        string name,
        double? actual,
        double required,
        int count,
        int minimumCount)
    {
        if (!actual.HasValue || count < minimumCount)
        {
            return new QualityGate(
                name,
                GateStatus.Inconclusive,
                actual,
                required,
                "<=",
                $"Need at least {minimumCount} cases; got {count}.");
        }
        return new QualityGate(
            name,
            actual <= required ? GateStatus.Passed : GateStatus.Failed,
            actual,
            required,
            "<=",
            $"Point estimate over {count} independent source clusters.");
    }

    private static double? Ratio(int numerator, int denominator)
        => denominator > 0 ? numerator / (double)denominator : null;

    /// <summary>
    /// Percentile over the clusters that produced an observation, with no censoring. Use together
    /// with the matching recall metric: recall reports how often the event happened, this reports how
    /// fast it was when it did. Keeping the two separate avoids charging one miss twice.
    /// </summary>
    private static double? ObservedPercentile(
        IReadOnlyList<ClusterObservation> expected,
        Func<ClusterObservation, double?> latency,
        double percentile)
    {
        double[] values = expected
            .Select(latency)
            .Where(value => value.HasValue)
            .Select(value => value!.Value)
            .OrderBy(value => value)
            .ToArray();
        if (values.Length == 0)
            return null;
        double position = percentile * (values.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = (int)Math.Ceiling(position);
        if (lower == upper)
            return values[lower];
        return values[lower] + (values[upper] - values[lower]) * (position - lower);
    }

    /// <summary>
    /// Number of expected clusters that produced no observation. Reported next to every latency
    /// percentile so it stays visible that the percentile is conditional on the event happening.
    /// </summary>
    private static int CensoredCount(
        IReadOnlyList<ClusterObservation> expected,
        Func<ClusterObservation, bool> observed)
        => expected.Count - expected.Count(observed);

    private static string SerBucket(double ser)
        => ser switch
        {
            <= -6 => "[-12,-6]dB",
            <= 0 => "(-6,0]dB",
            <= 6 => "(0,6]dB",
            _ => "(6,12]dB",
        };

    private sealed record ClusterObservation(
        string Id,
        EndToEndCaseResult[] Cases)
    {
        public bool HasError => Cases.Any(c => c.Error is not null);
        public bool ExpectHeard => Cases.Any(c => c.ExpectHeard);
        public bool ExpectDuck => Cases.Any(c => c.ExpectDuck);
        public bool ExpectAccepted => Cases.Any(c => c.ExpectAccepted);
        public bool ExpectPlaybackStop => Cases.Any(c => c.ExpectPlaybackStop);
        public bool ExpectPlaybackResume => Cases.Any(c => c.ExpectPlaybackResume);
        public bool HeardSuccess =>
            ExpectHeard &&
            Cases.Where(c => c.ExpectHeard)
                .All(c => c.Heard && !c.EarlyFalseTrigger);
        public bool DuckSuccess =>
            ExpectDuck &&
            Cases.Where(c => c.ExpectDuck)
                .All(c => c.PlaybackDucked && !c.EarlyFalseTrigger);
        public bool AcceptedSuccess =>
            ExpectAccepted &&
            Cases.Where(c => c.ExpectAccepted)
                .All(c => c.Accepted && !c.EarlyFalseTrigger);
        public bool StopSuccess =>
            ExpectPlaybackStop &&
            Cases.Where(c => c.ExpectPlaybackStop)
                .All(c => c.PlaybackStopped && !c.EarlyFalseTrigger);
        public bool AudibleStopSuccess =>
            StopSuccess &&
            Cases.Where(c => c.ExpectPlaybackStop)
                .All(c => c.AudibleStopLatencyMs.HasValue);
        public bool AudibleTailSuccess =>
            AudibleStopSuccess &&
            Cases.Where(c => c.ExpectPlaybackStop)
                .All(c => c.AudibleTailLatencyMs.HasValue);
        public bool ResumeSuccess =>
            ExpectPlaybackResume &&
            Cases.Where(c => c.ExpectPlaybackResume).All(c => c.PlaybackResumed);

        public double? MaxLatency(Func<EndToEndCaseResult, double?> selector)
        {
            if (Cases.Any(c => c.EarlyFalseTrigger))
                return null;
            double?[] values = Cases.Select(selector).ToArray();
            return values.All(value => value.HasValue)
                ? values.Max(value => value!.Value)
                : null;
        }
    }
}

internal static class EndToEndReportWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    public static (string JsonPath, string CsvPath) Write(
        EndToEndReport report,
        string? requestedPath)
    {
        string jsonPath = string.IsNullOrWhiteSpace(requestedPath)
            ? Path.Combine(
                Environment.CurrentDirectory,
                $"barge-in-e2e-{DateTime.Now:yyyyMMdd-HHmmss}.json")
            : Path.GetFullPath(requestedPath);
        if (!string.Equals(Path.GetExtension(jsonPath), ".json", StringComparison.OrdinalIgnoreCase))
            jsonPath += ".json";
        Directory.CreateDirectory(Path.GetDirectoryName(jsonPath)!);
        File.WriteAllText(
            jsonPath,
            JsonSerializer.Serialize(report, JsonOptions),
            new UTF8Encoding(false));

        string csvPath = Path.ChangeExtension(jsonPath, ".cases.csv");
        using var writer = new StreamWriter(csvPath, append: false, new UTF8Encoding(false));
        writer.WriteLine(
            "id,clusterId,corpus,scenario,geometryId,speakerPolicy,speechAssetId,expectedNearEnd," +
            "expectHeard,expectDuck,expectAccepted,expectPlaybackStop,expectPlaybackResume," +
            "heard,ducked,accepted,rejected,playbackStopped,playbackResumed,earlyFalseTrigger," +
            "triggerSource,decisionFrameCount,decisionAcceptedFrameCount,decisionEchoLikeFrameCount," +
            "decisionAcceptedDurationMs,decisionScoreDurationMs,decisionPeakScoreDurationMs," +
            "referenceText,recognizedText,cer,characterErrors,referenceCharacters,exactMatch," +
            "emptyRecognition,heardLatencyMs,heardLatencyFromSpeechOnsetMs," +
            "duckLatencyMs,audibleDuckLatencyMs,stopLatencyMs,audibleStopLatencyMs," +
            "audibleTailLatencyMs,drainCompleteLatencyMs,deviceTailLatencyMs," +
            "nearEndInsertionOnsetMs,nearEndSpeechOnsetMs,nearEndEligibleOnsetMs," +
            "nearEndEligible,nearEndActiveDurationMs,callbackSamples," +
            "serDb,snrDb,realizedSerDb,nearEndActiveLevel,echoActiveLevel,erlDb," +
            "speakerGain,clippedSampleRatio," +
            "echoDelayMs,clockDriftPpm,farEndNonlinear,renderGapStartMs," +
            "renderGapDurationMs,echoRirId,nearRirId," +
            "noiseAssetId,noiseKind,noiseCorpus,reason,error");
        foreach (EndToEndCaseResult c in report.Cases)
        {
            writer.WriteLine(string.Join(",",
                Csv(c.Id),
                Csv(c.ClusterId),
                Csv(c.Corpus),
                Csv(c.Scenario),
                Csv(c.GeometryId),
                Csv(c.SpeakerPolicy),
                Csv(c.SpeechAssetId ?? string.Empty),
                Bool(c.ExpectedNearEnd),
                Bool(c.ExpectHeard),
                Bool(c.ExpectDuck),
                Bool(c.ExpectAccepted),
                Bool(c.ExpectPlaybackStop),
                Bool(c.ExpectPlaybackResume),
                Bool(c.Heard),
                Bool(c.PlaybackDucked),
                Bool(c.Accepted),
                Bool(c.Rejected),
                Bool(c.PlaybackStopped),
                Bool(c.PlaybackResumed),
                Bool(c.EarlyFalseTrigger),
                Csv(c.TriggerSource ?? string.Empty),
                c.DecisionFrameCount,
                c.DecisionAcceptedFrameCount,
                c.DecisionEchoLikeFrameCount,
                Number(c.DecisionAcceptedDurationMs),
                Number(c.DecisionScoreDurationMs),
                Number(c.DecisionPeakScoreDurationMs),
                Csv(c.ReferenceText),
                Csv(c.RecognizedText),
                Number(c.CharacterErrorRate),
                c.CharacterErrors,
                c.ReferenceCharacters,
                Bool(c.ExactMatch),
                Bool(c.EmptyRecognition),
                Number(c.HeardLatencyMs),
                Number(c.HeardLatencyFromSpeechOnsetMs),
                Number(c.DuckLatencyMs),
                Number(c.AudibleDuckLatencyMs),
                Number(c.StopLatencyMs),
                Number(c.AudibleStopLatencyMs),
                Number(c.AudibleTailLatencyMs),
                Number(c.DrainCompleteLatencyMs),
                Number(c.DeviceTailLatencyMs),
                Number(c.NearEndInsertionOnsetMs),
                Number(c.NearEndSpeechOnsetMs),
                Number(c.NearEndEligibleOnsetMs),
                Bool(c.NearEndEligible),
                Number(c.NearEndActiveDurationMs),
                c.CallbackSamples,
                Number(c.SignalToEchoRatioDb),
                Number(c.SignalToNoiseRatioDb),
                Number(c.RealizedSignalToEchoRatioDb),
                Number(c.NearEndActiveLevel),
                Number(c.EchoActiveLevel),
                Number(c.EchoReturnLossDb),
                Number(c.SpeakerGain),
                Number(c.ClippedSampleRatio),
                c.EchoDelayMs,
                c.ClockDriftPpm,
                Bool(c.FarEndNonlinear),
                c.RenderGapStartMs?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
                c.RenderGapDurationMs,
                Csv(c.EchoRirId),
                Csv(c.NearRirId ?? string.Empty),
                Csv(c.NoiseAssetId ?? string.Empty),
                Csv(c.NoiseKind ?? string.Empty),
                Csv(c.NoiseCorpus ?? string.Empty),
                Csv(c.ValidationReason ?? string.Empty),
                Csv(c.Error ?? string.Empty)));
        }
        return (jsonPath, csvPath);
    }

    private static string Csv(string value)
        => $"\"{value.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";

    private static string Bool(bool value) => value ? "true" : "false";

    private static string Number(double value)
        => value.ToString("0.########", CultureInfo.InvariantCulture);

    private static string Number(double? value)
        => value?.ToString("0.########", CultureInfo.InvariantCulture) ?? string.Empty;
}
