using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Adapters;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static partial class SelfChecks
{
    public static void Run()
    {
        OracleFindsEligibleSpeechAfterGuard();
        OracleRoundsNonAlignedGuardUp();
        AggregateComputesExpectedRates();
        ZeroEventRateUsesPoissonUpperBound();
        MissesRightCensorLatency();
        EndToEndEarlyTriggersAreExcluded();
        EndToEndMetricsCountClusters();
        LegacyManifestStillLoads();
        PlaybackFifoHonorsGainAndClear();
        PartitionedFirMatchesOfflineConvolution();
        EvaluationErrorsFailEnforcedGates();
        DeviceCorpusNamesAreClassified();
        DeviceDirectoryAcceptsMicCFiles();
        DeviceFullbandHyphenatedScenariosAreDiscovered();
        MissingMobileCoverageIsInconclusive();
        PreparedTranscriptFormatsAreParsed();
        BenchmarkTimersFollowDatasetClock();
        AcousticWaitUsesInterruptionAdmissionLifecycle();
        ComparisonAllowsAlgorithmChangesAndCoversEverySliceMetric();
        ComparisonRejectsNewCaseRegressions();
        ComparisonEnforcesGateTransitions();
        ComparisonLoadsFrozenLegacyAlgorithmFieldsStrictly();
        ComparisonAllowsAlgorithmModelHashChangesOnly();
        ComparisonNormalizesLocalizedDotnetInfoButKeepsRuntimeIdentity();
        ComparisonIgnoresVolatileGpuTelemetryButKeepsGpuIdentity();
        ComparisonCliIsStrictAndDoesNotCaptureExistingModes();
    }

    private static void AcousticWaitUsesInterruptionAdmissionLifecycle()
    {
        var probe = new VoiceAssistantBenchmarkProbe();
        var admission = new MutableAdmissionProgress { IsSettled = false };
        probe.AttachInterruptionAdmissionView(admission);

        Assert(
            BargeInCorpusEvaluator.ShouldWaitForQueuedConfirmation(probe, heardPublished: false),
            "an admitted confirmation must keep the acoustic clock on the current callback");

        admission.IsSettled = true;
        Assert(
            !BargeInCorpusEvaluator.ShouldWaitForQueuedConfirmation(probe, heardPublished: false),
            "a rejected confirmation with no pending capture may release the acoustic clock");

        probe.ObserveControlPlaneProgress(
            new ActiveNearEndDiagnosticSnapshot(
                CandidateActive: false,
                CandidateAuthorized: false,
                CandidateDecisionEndSample: 0,
                MainAudioEndSample: 0,
                ActiveTurnId: 2,
                LastCandidateOutcome: "queued:confirmation",
                ConfirmationPending: false),
            VoiceBox.Assistant.Api.State.AssistantState.Speaking,
            isSpeakingWithoutPendingCandidate: false);
        Assert(
            BargeInCorpusEvaluator.ShouldWaitForQueuedConfirmation(probe, heardPublished: false),
            "an accepted capture must wait for its trailing Heard publication");
        Assert(
            !BargeInCorpusEvaluator.ShouldWaitForQueuedConfirmation(probe, heardPublished: true),
            "the Heard publication must release the acoustic clock");
    }

    private static void OracleFindsEligibleSpeechAfterGuard()
    {
        const int sampleRate = 16000;
        var audio = new float[sampleRate * 2];
        for (int i = sampleRate * 3 / 5; i < sampleRate * 6 / 5; i++)
            audio[i] = 0.1f;

        OracleActivity activity = OracleActivityAnalyzer.Analyze(
            audio,
            scale: 1,
            sampleRate,
            startupGuardMs: 500,
            minimumSpeechMs: 300);

        Assert(activity.Available, "oracle should be available");
        Assert(activity.Eligible, "oracle should find eligible speech");
        Assert(activity.EligibleOnsetSample == sampleRate * 3 / 5, "oracle onset should be 600 ms");
    }

    private static void OracleRoundsNonAlignedGuardUp()
    {
        const int sampleRate = 16000;
        var audio = new float[sampleRate];
        for (int i = sampleRate / 2; i < sampleRate * 3 / 5; i++)
            audio[i] = 0.1f;

        OracleActivity activity = OracleActivityAnalyzer.Analyze(
            audio,
            scale: 1,
            sampleRate,
            startupGuardMs: 501,
            minimumSpeechMs: 20);

        Assert(activity.Eligible, "oracle should find speech after a non-aligned guard");
        Assert(
            activity.EligibleOnsetSample == sampleRate * 520 / 1000,
            "non-aligned startup guard must round up to the next 20 ms frame");
    }

    private static void AggregateComputesExpectedRates()
    {
        CaseResult[] cases =
        [
            Result("tp", expected: true, detected: true, tp: true),
            Result("fn", expected: true, detected: false, fn: true),
            Result("tn", expected: false, detected: false),
            Result("fp", expected: false, detected: true, fp: true),
        ];
        MetricSlice metrics = MetricAggregator.Aggregate("self-test", cases);
        Assert(metrics.Recall == 0.5, "recall should be 0.5");
        Assert(metrics.Specificity == 0.5, "specificity should be 0.5");
        Assert(metrics.FalsePositive == 1, "false positive count should be one");
        Assert(metrics.RecallLower95 is > 0 and < 0.5, "recall lower confidence bound should be below the point estimate");
        Assert(metrics.RecallUpper95 is > 0.5 and < 1, "recall upper confidence bound should be above the point estimate");
    }

    private static void ZeroEventRateUsesPoissonUpperBound()
    {
        CaseResult[] cases =
        [
            Result("tn", expected: false, detected: false, durationMs: 3_600_000),
        ];
        MetricSlice metrics = MetricAggregator.Aggregate("poisson", cases);
        Assert(
            metrics.FalseTriggersPerHourUpper95 is > 2.995 and < 2.996,
            "zero events over one hour should have an exact one-sided 95% upper rate near 2.996/hour");

        var options = new EvaluationOptions
        {
            MinimumGateNegativeCount = 1,
        };
        QualityGate gate = MetricAggregator.EvaluateGates(cases, options)
            .Single(g => g.Name.EndsWith("/farend-false-triggers-per-hour", StringComparison.Ordinal));
        Assert(gate.Status == GateStatus.Inconclusive, "uncertain zero-event rate should not pass");
    }

    private static void MissesRightCensorLatency()
    {
        var cases = new List<CaseResult>();
        for (int i = 0; i < 19; i++)
        {
            cases.Add(Result(
                $"tp-{i}",
                expected: true,
                detected: true,
                tp: true,
                eligibleLatencyMs: 100));
        }
        cases.Add(Result("fn", expected: true, detected: false, fn: true));

        MetricSlice metrics = MetricAggregator.Aggregate("censored", cases);
        Assert(metrics.OraclePositive == 20, "all positive self-check cases should have an oracle");
        Assert(metrics.LatencyCount == 19, "nineteen latency values should be observed");
        Assert(metrics.LatencyCensored == 1, "one missed detection should be right-censored");
        Assert(metrics.LatencyP95Ms is null, "a censored top five percent should make empirical P95 unbounded");

        QualityGate gate = MetricAggregator.EvaluateGates(cases, new EvaluationOptions())
            .Single(g => g.Name.EndsWith("/eligible-onset-to-decision-p95-ms", StringComparison.Ordinal));
        Assert(gate.Status == GateStatus.Failed, "right-censored P95 should fail the latency gate");
    }

    private static void EndToEndEarlyTriggersAreExcluded()
    {
        EndToEndCaseResult[] cases =
        [
            EndToEndResult(
                "early",
                heard: true,
                accepted: true,
                earlyFalseTrigger: true,
                heardLatencyMs: -50,
                stopLatencyMs: 500),
            EndToEndResult(
                "valid",
                heard: true,
                accepted: true,
                heardLatencyMs: 100,
                stopLatencyMs: 600),
        ];

        EndToEndSummary summary = EndToEndMetrics.Aggregate(cases);
        Assert(summary.EarlyFalseTrigger == 1, "early trigger count should be reported");
        Assert(summary.HeardPositive == 1, "early Heard must not count as a positive recall");
        Assert(summary.AcceptedPositive == 1, "early Accepted must not count as a positive recall");
        Assert(summary.HeardRecall == 0.5, "early Heard should reduce recall");
        Assert(summary.AcceptedRecall == 0.5, "early Accepted should reduce recall");
        Assert(summary.HeardLatencyCensored == 1, "early Heard must right-censor latency");
        Assert(summary.HeardLatencyP50Ms == 100, "conditional Heard latency must exclude early triggers");
        Assert(summary.HeardLatencyP95Ms == 100, "conditional Heard P95 must preserve the valid observation");
        Assert(summary.StopLatencyCensored == 1, "early stop must right-censor stop latency");
        Assert(summary.StopLatencyP95Ms == 600, "conditional stop latency must exclude early triggers");
    }

    private static void PreparedTranscriptFormatsAreParsed()
    {
        const string utteranceId = "X0000000021_240514196_S00041";
        string wenet =
            $"{utteranceId}\t跟随的炮船传回了回波。\r\n" +
            "\t[[440, 560], [560, 800]]";
        Assert(
            PreparedSpeechTranscript.ParseWenet(wenet, utteranceId) ==
            "跟随的炮船传回了回波。",
            "Wenet transcript parser should remove the utterance id and timing annotations");

        string root = Path.Combine(
            Path.GetTempPath(),
            "voicebox-transcript-self-check-" + Guid.NewGuid().ToString("N"));
        string data = Path.Combine(root, "data");
        string test = Path.Combine(root, "test");
        Directory.CreateDirectory(data);
        Directory.CreateDirectory(test);
        try
        {
            File.WriteAllText(
                Path.Combine(data, "D11_915.wav.trn"),
                "除了 付给 教师 们 工资 外\r\nchu2 le5");
            string pointer = Path.Combine(test, "D11_915.wav.trn");
            File.WriteAllText(pointer, "../data/D11_915.wav.trn");
            Assert(
                PreparedSpeechTranscript.ReadThchs30(pointer) ==
                "除了付给教师们工资外",
                "THCHS30 transcript parser should follow extracted symlink placeholders");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void BenchmarkTimersFollowDatasetClock()
    {
        var time = new BenchmarkTimeProvider();
        Task delay = Task.Delay(TimeSpan.FromMilliseconds(100), time);
        time.AdvanceSamples(99, sampleRate: 1000, channels: 1);
        Assert(!delay.IsCompleted, "benchmark delay should not complete before its virtual due time");
        time.AdvanceSamples(1, sampleRate: 1000, channels: 1);
        Assert(
            delay.IsCompletedSuccessfully,
            "benchmark delay should complete when dataset time reaches its due time");
    }

    private static void EndToEndMetricsCountClusters()
    {
        EndToEndCaseResult[] cases =
        [
            EndToEndResult("variant-a", heard: true, accepted: true) with
            {
                ClusterId = "speaker-1",
            },
            EndToEndResult("variant-b", heard: false, accepted: false) with
            {
                ClusterId = "speaker-1",
            },
        ];
        EndToEndSummary summary = EndToEndMetrics.Aggregate(cases);
        Assert(summary.Total == 2, "raw end-to-end case count should be preserved");
        Assert(summary.ClusterCount == 1, "two variants of one speaker should form one cluster");
        Assert(summary.Positive == 1, "the positive denominator must count source clusters");
        Assert(summary.HeardRecall == 0, "a failed variant should fail its source cluster");
    }

    private static void LegacyManifestStillLoads()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            $"voicebox-manifest-self-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllText(
                Path.Combine(root, EvaluationDatasetManifest.FileName),
                """
                {
                  "schemaVersion": "1.0",
                  "datasetId": "legacy",
                  "createdUtc": "2026-01-01T00:00:00Z",
                  "seed": 1,
                  "materialization": "copy",
                  "sourceRoots": {},
                  "acousticCases": [],
                  "speechAssets": [],
                  "rirAssets": [],
                  "noiseAssets": [],
                  "endToEndCases": []
                }
                """);
            EvaluationDatasetManifest manifest = EvaluationDatasetManifest.Load(root);
            Assert(manifest.DatasetId == "legacy", "schema 1.0 manifests should remain readable");
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    private static void PlaybackFifoHonorsGainAndClear()
    {
        using var ingress = new BenchmarkDuplexAudioIngress(
            new PassthroughCaptureProcessor(),
            TimeProvider.System)
        {
            PlaybackGainRampMs = 0,
        };
        ingress.StartAsync().GetAwaiter().GetResult();
        Assert(
            ingress.TryEnqueue([1f, 0.5f, -0.5f, -1f], out int dropped) &&
            dropped == 0,
            "playback samples should enter the benchmark FIFO");
        ingress.SetPlaybackGain(0.5f);
        var first = new float[2];
        ingress.ReadPlayback(first);
        Assert(
            first.SequenceEqual([0.5f, 0.25f]),
            "playback gain must affect samples read from the FIFO");
        ingress.ClearQueue();
        var afterClear = new float[2];
        ingress.ReadPlayback(afterClear);
        Assert(
            afterClear.All(sample => sample == 0),
            "ClearQueue must silence all future render samples");
    }

    private static void PartitionedFirMatchesOfflineConvolution()
    {
        const int blockSize = 3;
        float[] signal = [0.4f, -0.2f, 0.1f, 0.8f, 0.3f, -0.5f, 0.2f];
        float[] impulse = [0.7f, -0.1f, 0.25f, 0.05f, -0.02f];
        float[] expected = AcousticSceneRenderer.Convolve(signal, impulse);
        var filter = new AcousticSceneRenderer.PartitionedFirFilter(
            impulse,
            blockSize);
        int outputLength = ((expected.Length + blockSize - 1) / blockSize) * blockSize;
        var actual = new float[outputLength];
        for (int offset = 0; offset < outputLength; offset += blockSize)
        {
            var input = new float[blockSize];
            int count = Math.Min(blockSize, Math.Max(0, signal.Length - offset));
            if (count > 0)
                signal.AsSpan(offset, count).CopyTo(input);
            filter.Process(input, actual.AsSpan(offset, blockSize));
        }
        for (int i = 0; i < expected.Length; i++)
        {
            Assert(
                Math.Abs(expected[i] - actual[i]) < 1e-5,
                $"partitioned FIR sample {i} should match offline convolution");
        }
    }

    private static void EvaluationErrorsFailEnforcedGates()
    {
        CaseResult[] cases =
        [
            Result("error", expected: true, detected: false, fn: true) with
            {
                Error = "InvalidDataException: broken test asset",
            },
        ];

        QualityGate gate = MetricAggregator.EvaluateGates(
                cases,
                new EvaluationOptions
                {
                    MinimumGatePositiveCount = 1,
                    MinimumGateNegativeCount = 1,
                    MinimumGateLatencyCount = 1,
                })
            .Single(g => g.Name == "evaluation/errors");
        Assert(gate.Status == GateStatus.Failed, "case errors must fail enforced evaluation gates");
    }

    private static void DeviceCorpusNamesAreClassified()
    {
        bool classified = DatasetCatalog.TryClassifyDeviceMicrophoneName(
            "device-123_doubletalk_with_movement_mic.wav",
            out string scenario,
            out bool expectedNearEnd,
            out string session);
        Assert(classified, "device double-talk filename should be classified");
        Assert(scenario == "doubletalk_with_movement", "movement scenario should be preserved");
        Assert(expectedNearEnd, "double-talk should be a positive case");
        Assert(session == "device-123", "anonymous device session should be extracted");

        classified = DatasetCatalog.TryClassifyDeviceMicrophoneName(
            "device-456_farend_singletalk_mic.wav",
            out scenario,
            out expectedNearEnd,
            out session);
        Assert(classified, "device far-end filename should be classified");
        Assert(scenario == "farend_singletalk", "far-end scenario should be preserved");
        Assert(!expectedNearEnd, "far-end single-talk should be a negative case");
        Assert(session == "device-456", "far-end device session should be extracted");

        classified = DatasetCatalog.TryClassifyDeviceMicrophoneName(
            "device-789_doubletalk-with-movement_mic.wav",
            out scenario,
            out expectedNearEnd,
            out session);
        Assert(classified, "hyphenated fullband double-talk filename should be classified");
        Assert(scenario == "doubletalk_with_movement", "hyphenated scenario should be normalized");
        Assert(expectedNearEnd, "hyphenated double-talk should be a positive case");
        Assert(session == "device-789", "hyphenated fullband session should be extracted");
    }

    private static void DeviceFullbandHyphenatedScenariosAreDiscovered()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            $"voicebox-aec-self-test-{Guid.NewGuid():N}");
        string positiveDirectory = Path.Combine(
            root,
            "datasets",
            "fullband",
            "doubletalk-with-movement");
        string negativeDirectory = Path.Combine(
            root,
            "datasets",
            "fullband",
            "farend-singletalk");
        Directory.CreateDirectory(positiveDirectory);
        Directory.CreateDirectory(negativeDirectory);
        try
        {
            foreach ((string directory, string session) in new[]
                     {
                         (positiveDirectory, "positive-session"),
                         (negativeDirectory, "negative-session"),
                     })
            {
                File.WriteAllBytes(Path.Combine(directory, $"{session}_mic.wav"), []);
                File.WriteAllBytes(Path.Combine(directory, $"{session}_lpb.wav"), []);
            }

            var options = new EvaluationOptions
            {
                DatasetRoot = root,
                Suite = EvaluationSuite.DeviceSample,
            };
            IReadOnlyList<EvaluationCase> cases = DatasetCatalog.Discover(options);
            Assert(cases.Count == 2, "fullband hyphenated scenario directories should be discovered");
            Assert(
                cases.Any(item =>
                    item.Scenario == "doubletalk_with_movement" &&
                    item.ExpectedNearEnd),
                "fullband movement double-talk should be a positive case");
            Assert(
                cases.Any(item =>
                    item.Scenario == "farend_singletalk" &&
                    !item.ExpectedNearEnd),
                "fullband far-end single-talk should be a negative case");
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    private static void DeviceDirectoryAcceptsMicCFiles()
    {
        string root = Path.Combine(
            Path.GetTempPath(),
            $"voicebox-aec-self-test-{Guid.NewGuid():N}");
        string directory = Path.Combine(root, "datasets", "real");
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllBytes(
                Path.Combine(directory, "session_doubletalk_mic_c.wav"),
                []);
            File.WriteAllBytes(
                Path.Combine(directory, "session_doubletalk_lpb.wav"),
                []);

            var options = new EvaluationOptions
            {
                DatasetRoot = root,
                Suite = EvaluationSuite.DeviceSample,
                LimitPerStratum = 1,
            };
            IReadOnlyList<EvaluationCase> cases = DatasetCatalog.Discover(options);
            Assert(cases.Count == 1, "device discovery should accept _mic_c.wav recordings");
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    private static void MissingMobileCoverageIsInconclusive()
    {
        CaseResult[] cases =
        [
            Result(
                "windows-positive",
                expected: true,
                detected: true,
                tp: true,
                deviceClass: DatasetCatalog.WindowsPcDeviceClass),
            Result(
                "windows-negative",
                expected: false,
                detected: false,
                deviceClass: DatasetCatalog.WindowsPcDeviceClass),
        ];
        var options = new EvaluationOptions
        {
            Suite = EvaluationSuite.DeviceSample,
            MinimumGatePositiveCount = 1,
            MinimumGateNegativeCount = 1,
        };
        IReadOnlyList<QualityGate> gates = MetricAggregator.EvaluateGates(cases, options);
        QualityGate windows = gates.Single(g =>
            g.Name.EndsWith(
                $"/device-coverage:{DatasetCatalog.WindowsPcDeviceClass}",
                StringComparison.Ordinal));
        QualityGate mobile = gates.Single(g =>
            g.Name.EndsWith(
                $"/device-coverage:{DatasetCatalog.AndroidMobileDeviceClass}",
                StringComparison.Ordinal));
        Assert(windows.Status == GateStatus.Passed, "present Windows device coverage should pass");
        Assert(mobile.Status == GateStatus.Inconclusive, "missing mobile coverage must be inconclusive");
    }

    private static CaseResult Result(
        string id,
        bool expected,
        bool detected,
        bool tp = false,
        bool fn = false,
        bool fp = false,
        double durationMs = 1000,
        double? eligibleLatencyMs = null,
        string? deviceClass = null)
        => new()
        {
            Id = id,
            Corpus = "self-test",
            Scenario = "self-test",
            DeviceClass = deviceClass,
            DeviceSession = deviceClass is null ? null : id,
            ClusterId = deviceClass is null ? id : $"{deviceClass}/{id}",
            ExpectedNearEnd = expected,
            CallbackSamples = 256,
            SourceSampleRate = 16000,
            EvaluationSampleRate = 16000,
            DurationMs = durationMs,
            EvaluatedDurationMs = durationMs,
            Resampled = false,
            Detected = detected,
            DetectionTimeMs = detected ? 500 : null,
            DecisionReason = "self-test",
            DecisionFrameCount = 1,
            DecisionAcceptedFrameCount = tp ? 1 : 0,
            DecisionEchoLikeFrameCount = 0,
            DecisionAcceptedDurationMs = tp ? 300 : 0,
            DecisionScoreDurationMs = tp ? 300 : 0,
            DecisionPeakScoreDurationMs = tp ? 300 : 0,
            PeakNearEndProbability = tp ? 1 : 0,
            PeakSmoothedNearEndProbability = tp ? 1 : 0,
            Oracle = expected
                ? new OracleActivity(true, true, 0, 0, 16000, 1000, 0.001)
                : OracleActivity.Unavailable,
            EligibleLatencyMs = tp ? eligibleLatencyMs ?? 500 : null,
            EarlyFalseTrigger = false,
            TruePositive = tp,
            FalseNegative = fn,
            FalsePositive = fp,
            SignalToEchoRatioDb = null,
            FarEndNonlinear = null,
            FarEndNoisy = null,
            NearEndNoisy = null,
        };

    private static EndToEndCaseResult EndToEndResult(
        string id,
        bool heard,
        bool accepted,
        bool earlyFalseTrigger = false,
        double? heardLatencyMs = null,
        double? stopLatencyMs = null)
        => new()
        {
            Id = id,
            ClusterId = id,
            Corpus = "self-test",
            SpeechAssetId = id,
            ExpectedNearEnd = true,
            Scenario = "valid-command",
            GeometryId = "room/mic/position",
            SpeakerPolicy = "any-speaker",
            ExpectHeard = true,
            ExpectDuck = true,
            ExpectAccepted = true,
            ExpectPlaybackStop = true,
            ExpectPlaybackResume = false,
            ReferenceText = "测试",
            RecognizedText = "测试",
            Heard = heard,
            EarlyFalseTrigger = earlyFalseTrigger,
            Accepted = accepted,
            Rejected = false,
            PlaybackDucked = heard,
            PlaybackStopped = accepted,
            PlaybackResumed = false,
            ValidationReason = accepted ? "recognized" : null,
            CallbackSamples = 160,
            AudioDurationMs = 1000,
            HeardLatencyMs = heardLatencyMs,
            DuckLatencyMs = null,
            StopLatencyMs = stopLatencyMs,
            AudibleStopLatencyMs = stopLatencyMs,
            AudibleTailLatencyMs = stopLatencyMs,
            CharacterErrors = 0,
            ReferenceCharacters = 2,
            CharacterErrorRate = 0,
            ExactMatch = true,
            EmptyRecognition = false,
            SignalToEchoRatioDb = 0,
            SignalToNoiseRatioDb = 20,
            EchoDelayMs = 40,
            ClockDriftPpm = 0,
            FarEndNonlinear = false,
            EchoRirId = "rir",
            NearRirId = "near-rir",
            NoiseAssetId = null,
        };

    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException($"Self-check failed: {message}.");
    }

    private sealed class PassthroughCaptureProcessor : IAcousticEchoCancellationService
    {
        public int DelaySamples => 0;

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate)
        {
        }

        public void Reset()
        {
        }

        public void Dispose()
        {
        }
    }

    private sealed class MutableAdmissionProgress : IInterruptionAdmissionProgressView
    {
        public bool IsSettled { get; set; }

        public bool HasSettledAllInterruptionAdmissions => IsSettled;
    }
}
