using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using VoiceBox.APM.WebRtc;
using VoiceBox.APM.WebRtc.Models;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Adapters;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed class BargeInCorpusEvaluator : IDisposable
{
    private const int EarlyTriggerToleranceMs = 100;

    private readonly EvaluationOptions _options;
    private readonly WebRtcApmService _mainAec;
    private readonly BenchmarkClockedAecService _clockedMainAec;
    private readonly BenchmarkTimeProvider _timeProvider;
    private readonly BenchmarkDuplexAudioIngress _audio;
    private readonly BenchmarkVadService _vad;
    private readonly BenchmarkAsrService _asr;
    private readonly BenchmarkLlmService _llm;
    private readonly BenchmarkTtsService _tts;
    private readonly AudioServiceManager _services;
    private bool _disposed;

    public BargeInCorpusEvaluator(EvaluationOptions options)
    {
        _options = options;
        _mainAec = new WebRtcApmService(Options.Create(new WebRtcApmOptions
        {
            SampleRate = EvaluationOptions.TargetSampleRate,
            ChannelCount = 1,
            EnableHighPassFilter = true,
            EnableEchoCanceller = true,
            EnableNoiseSuppression = options.EnableNoiseSuppression,
            EnableAgc2 = false,
        }));
        _timeProvider = new BenchmarkTimeProvider();
        _audio = new BenchmarkDuplexAudioIngress(
            _mainAec,
            _timeProvider,
            EvaluationOptions.TargetSampleRate,
            channels: 1);
        _vad = new BenchmarkVadService();
        _asr = new BenchmarkAsrService();
        _llm = new BenchmarkLlmService();
        _tts = new BenchmarkTtsService();
        _clockedMainAec = new BenchmarkClockedAecService(_mainAec, _mainAec, _timeProvider);

        _services = (AudioServiceManager)new AudioServiceManager()
            .AddService<IAudioDuplexService>(services =>
            {
                services.AddSingleton<IAudioDuplexService>(_audio);
                services.AddSingleton<IAudioRecordService>(_audio);
                services.AddSingleton<IAudioPlayService>(_audio);
                services.AddSingleton<IAcousticEchoCancellationService>(_clockedMainAec);
                services.AddSingleton<IVoiceActivityDetectService>(_vad);
                services.AddSingleton<IASRService>(_asr);
                services.AddSingleton<IStreamLLMService>(_llm);
                services.AddSingleton<IStreamTTSService>(_tts);
            })
            .Build();
    }

    public async Task<CaseResult> EvaluateAsync(
        EvaluationCase testCase,
        int callbackSamples,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (callbackSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(callbackSamples));

        WaveAudio microphone = WaveAudioLoader.LoadMono(
            testCase.MicrophonePath,
            EvaluationOptions.TargetSampleRate);
        WaveAudio renderAudio = WaveAudioLoader.LoadMono(
            testCase.RenderPath,
            EvaluationOptions.TargetSampleRate);
        WaveAudio? nearEnd = testCase.NearEndReferencePath is null
            ? null
            : WaveAudioLoader.LoadMono(testCase.NearEndReferencePath, EvaluationOptions.TargetSampleRate);

        OracleActivity oracle = nearEnd is null
            ? OracleActivity.Unavailable
            : OracleActivityAnalyzer.Analyze(
                nearEnd.Samples,
                testCase.NearEndScale,
                EvaluationOptions.TargetSampleRate,
                _options.PlaybackStartupGuardMs,
                _options.MinDurationMs);

        _timeProvider.Reset();
        _tts.BeginCase();
        try
        {
            var probe = new VoiceAssistantBenchmarkProbe();
            probe.AttachCapturePosition(_audio);
            VoiceAssistantOptions assistantOptions = CreateAssistantOptions();
            VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
                .UseOptions(assistantOptions)
                .UseAudio(_audio)
                .UseTurnBoundary(_vad)
                .UseSpeechRecognition(_asr)
                .UseResponseGeneration(_llm)
                .UseSpeechSynthesis(_tts)
                .AddAcousticDiagnostics(_clockedMainAec)
                .AddBargeIn(
                    assistantOptions.BargeIn,
                    new WebRtcAec3DelayEstimatorFactory());
            using var assistant = builder.BuildForBenchmark(_timeProvider, probe);

            bool detected = false;
            BargeInValidationEventArgs? heard = null;
            assistant.BargeInValidation += OnBargeInValidation;
            try
            {
                await assistant.StartAsync(cancellationToken).ConfigureAwait(false);
                _vad.EmitBootstrapUtterance();
                await WaitUntilAsync(
                    () => assistant.State == AssistantState.Speaking && _audio.IsPlaying,
                    cancellationToken).ConfigureAwait(false);

                int sampleRate = EvaluationOptions.TargetSampleRate;
                int totalSamples = Math.Max(microphone.Samples.Length, renderAudio.Samples.Length);
                double durationMs = totalSamples * 1000.0 / sampleRate;
                var rawBuffer = new float[callbackSamples];
                var renderBuffer = new float[callbackSamples];
                long evaluatedSamples = 0;

                for (int offset = 0; offset < totalSamples; offset += callbackSamples)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int length = Math.Min(callbackSamples, totalSamples - offset);
                    Span<float> raw = rawBuffer.AsSpan(0, length);
                    Span<float> render = renderBuffer.AsSpan(0, length);
                    raw.Clear();
                    render.Clear();
                    CopyAvailable(microphone.Samples, offset, raw);
                    CopyAvailable(renderAudio.Samples, offset, render);

                    _timeProvider.AdvanceSamples(length, sampleRate, channels: 1);
                    _audio.PushFrame(raw, render);
                    await AllowQueuedConfirmationToReachControlLoopAsync(
                        probe,
                        () => detected,
                        TimeSpan.FromSeconds((double)length / sampleRate),
                        cancellationToken).ConfigureAwait(false);
                    evaluatedSamples = offset + length;
                    if (detected)
                        break;
                }

                for (int drain = 0;
                     drain < 2 && !detected && _audio.CaptureProcessorDelaySamples > 0;
                     drain++)
                {
                    _timeProvider.AdvanceSamples(
                        _audio.CaptureProcessorDelaySamples,
                        sampleRate,
                        channels: 1);
                    _audio.FlushCaptureProcessorDelay();
                }

                VoiceAssistantBenchmarkSnapshot snapshot = probe.GetSnapshot();
                BargeInDecision decision = snapshot.Decision;
                if (heard is not null)
                {
                    decision = decision with
                    {
                        FrameCount = heard.AudioFrameCount,
                        AcceptedFrameCount = heard.AudioAcceptedFrameCount,
                        EchoLikeFrameCount = heard.AudioEchoLikeFrameCount,
                        AcceptedDurationMs = heard.AudioAcceptedDurationMs,
                        ScoreDurationMs = heard.AudioScoreDurationMs,
                        PeakScoreDurationMs = heard.AudioPeakScoreDurationMs,
                    };
                }

                bool gradeablePositive =
                    testCase.ExpectedNearEnd &&
                    (!oracle.Available || oracle.Eligible);
                bool earlyFalseTrigger =
                    detected &&
                    oracle.Available &&
                    oracle.Eligible &&
                    evaluatedSamples <
                    oracle.EligibleOnsetSample -
                    checked(EarlyTriggerToleranceMs * sampleRate / 1000);
                bool truePositive = gradeablePositive && detected && !earlyFalseTrigger;
                bool falseNegative = gradeablePositive && !truePositive;
                bool falsePositive = !testCase.ExpectedNearEnd && detected;
                double? eligibleLatencyMs =
                    truePositive && oracle.Available && oracle.Eligible
                        ? (evaluatedSamples - oracle.EligibleOnsetSample) * 1000.0 / sampleRate
                        : null;

                return new CaseResult
                {
                    Id = testCase.Id,
                    Corpus = testCase.Corpus,
                    Scenario = testCase.Scenario,
                    DeviceClass = testCase.DeviceClass,
                    DeviceSession = testCase.DeviceSession,
                    ClusterId = testCase.ClusterId ?? testCase.DeviceSession ?? testCase.Id,
                    ExpectedNearEnd = testCase.ExpectedNearEnd,
                    CallbackSamples = callbackSamples,
                    SourceSampleRate = microphone.SourceSampleRate,
                    EvaluationSampleRate = sampleRate,
                    DurationMs = durationMs,
                    EvaluatedDurationMs = evaluatedSamples * 1000.0 / sampleRate,
                    Resampled = microphone.Resampled || renderAudio.Resampled || nearEnd?.Resampled == true,
                    Detected = detected,
                    DetectionTimeMs = detected ? evaluatedSamples * 1000.0 / sampleRate : null,
                    DecisionReason = detected
                        ? "confirmed-nearend-window"
                        : decision.Reason ?? "none",
                    DecisionFrameCount = decision.FrameCount,
                    DecisionAcceptedFrameCount = decision.AcceptedFrameCount,
                    DecisionEchoLikeFrameCount = decision.EchoLikeFrameCount,
                    DecisionAcceptedDurationMs = decision.AcceptedDurationMs,
                    DecisionScoreDurationMs = decision.ScoreDurationMs,
                    DecisionPeakScoreDurationMs = decision.PeakScoreDurationMs,
                    PeakNearEndProbability = snapshot.PeakNearEndProbability,
                    PeakSmoothedNearEndProbability = snapshot.PeakSmoothedNearEndProbability,
                    Oracle = oracle,
                    EligibleLatencyMs = eligibleLatencyMs,
                    EarlyFalseTrigger = earlyFalseTrigger,
                    TruePositive = truePositive,
                    FalseNegative = falseNegative,
                    FalsePositive = falsePositive,
                    SignalToEchoRatioDb = testCase.SignalToEchoRatioDb,
                    FarEndNonlinear = testCase.FarEndNonlinear,
                    FarEndNoisy = testCase.FarEndNoisy,
                    NearEndNoisy = testCase.NearEndNoisy,
                    CoordinatorDiagnostic =
                        $"state={snapshot.AssistantState};" +
                        $"speakingReady={snapshot.IsSpeakingWithoutPendingCandidate};" +
                        $"playback={snapshot.PlaybackActive};" +
                        $"render={snapshot.RenderActive};" +
                        $"candidate={snapshot.Coordinator.CandidateActive};" +
                        $"authorized={snapshot.Coordinator.CandidateAuthorized};" +
                        $"decisionEnd={snapshot.Coordinator.CandidateDecisionEndSample};" +
                        $"mainEnd={snapshot.Coordinator.MainAudioEndSample};" +
                        $"turn={snapshot.Coordinator.ActiveTurnId};" +
                        $"outcome={snapshot.Coordinator.LastCandidateOutcome};" +
                        $"detectorObserved={snapshot.Detector.HasObservation};" +
                        $"detectorNearEnd={snapshot.Detector.LatestNearEndSpeech};" +
                        $"detectorProbability={snapshot.Detector.LatestProbability:F3};" +
                        $"detectorSmoothed={snapshot.Detector.LatestSmoothedProbability:F3};" +
                        $"alignmentReady={snapshot.Detector.AlignmentReady};" +
                        $"delayMs={snapshot.Detector.EstimatedDelayMs:F2};" +
                        $"driftPpm={snapshot.Detector.DriftPpm:F2};" +
                        $"alignmentConfidence={snapshot.Detector.AlignmentConfidence:F3};" +
                        $"alignmentGeneration={snapshot.Detector.AlignmentGeneration};" +
                        $"renderEpoch={snapshot.Detector.StreamEpoch};" +
                        $"renderRange={snapshot.Detector.LastStartSample}-{snapshot.Detector.LastEndSampleExclusive};" +
                        $"renderStatus={snapshot.Detector.RenderStatus};" +
                        $"detectorFaulted={snapshot.Detector.IsFaulted};" +
                        $"obs={snapshot.Detector.ObservationCount};" +
                        $"nearEndObs={snapshot.Detector.NearEndObservationCount};" +
                        $"postAecObs={snapshot.Detector.PostAecEvidenceCount};" +
                        $"echoLikeObs={snapshot.Detector.EchoLikeObservationCount};" +
                        $"peakCaptureRms={snapshot.Detector.PeakCaptureRms:F5};" +
                        $"peakPostAecRms={snapshot.Detector.PeakPostAecRms:F5};" +
                        $"minPostAecRatio={snapshot.Detector.MinimumPostAecResidualRatio:F3};" +
                        $"maxAbsInputRenderCorr={snapshot.Detector.MaximumAbsoluteRenderCorrelation:F3};" +
                        $"lastPostAecRatio={snapshot.Detector.LastPostAecResidualRatio:F3};" +
                        $"lastInputRenderCorr={snapshot.Detector.LastRenderCorrelation:F3};" +
                        $"lastPostAecRenderCorr={snapshot.Detector.LastPostAecRenderCorrelation:F3};" +
                        $"lastEchoLike={snapshot.Detector.LastObservationEchoLike};" +
                        $"localDelayObs={snapshot.Detector.LocalDelayEstimateCount};" +
                        $"localRefinedObs={snapshot.Detector.LocalRefinedEstimateCount};" +
                        $"lastLocalDelayMs={snapshot.Detector.LastLocalDelayMs:F3};" +
                        $"lastSelectedDelayMs={snapshot.Detector.LastSelectedDelayMs:F3};" +
                        $"lastSelectedLocal={snapshot.Detector.LastSelectedDelayWasLocal};" +
                        $"correlationFallbackCount={snapshot.Detector.CorrelationFallbackCount};" +
                        $"lastCorrelationFallbackDelayMs={snapshot.Detector.LastCorrelationFallbackDelayMs:F3};" +
                        $"minCorrelationFallbackDelayMs={snapshot.Detector.MinimumCorrelationFallbackDelayMs:F3};" +
                        $"maxCorrelationFallbackDelayMs={snapshot.Detector.MaximumCorrelationFallbackDelayMs:F3};" +
                        $"echoProfileReliability={snapshot.Detector.EchoProfileReliability:F3};" +
                        $"trustedEchoDurationMs={snapshot.Detector.TrustedEchoDurationMs:F1};" +
                        $"relevantEchoDurationMs={snapshot.Detector.RelevantEchoDurationMs:F1};" +
                        $"predictedEchoCaptureRms={snapshot.Detector.PredictedEchoCaptureRms:F5};" +
                        $"predictedEchoPostAecRms={snapshot.Detector.PredictedEchoPostAecRms:F5};" +
                        $"predictedEchoResidualRatio={snapshot.Detector.PredictedEchoResidualRatio:F3};" +
                        $"trustedEchoObs={snapshot.Detector.TrustedEchoObservationCount};" +
                        $"preservedNearEndObs={snapshot.Detector.PreservedNearEndObservationCount};" +
                        $"lastPreservedNearEnd={snapshot.Detector.LastObservationPreservedNearEnd};" +
                        $"pathGuardActive={snapshot.Detector.AcousticPathChangeGuardActive};" +
                        $"pathGuardRemainingMs={snapshot.Detector.AcousticPathChangeGuardRemainingMs:F1};" +
                        $"pathChangeCount={snapshot.Detector.AcousticPathChangeCount};" +
                        $"lastAcousticPathChange={snapshot.Detector.LastObservationAcousticPathChange};" +
                        $"lastPathChangeGuarded={snapshot.Detector.LastObservationPathChangeGuarded};" +
                        $"lastPathChangeReason={snapshot.Detector.LastAcousticPathChangeReason};" +
                        $"decisionPostAecCount={decision.PostAecEvidenceFrameCount};" +
                        $"decisionMinPostAecRatio={decision.MinPostAecResidualRatio:F3};" +
                        $"decisionMaxPostAecRatio={decision.MaxPostAecResidualRatio:F3};" +
                        $"decisionMeanPostAecRatio={decision.MeanPostAecResidualRatio:F3};" +
                        $"decisionMeanAbsInputRenderCorr={decision.MeanAbsoluteRenderCorrelation:F3};" +
                        $"decisionMeanAbsPostAecRenderCorr={decision.MeanAbsolutePostAecRenderCorrelation:F3}",
                };
            }
            finally
            {
                _tts.ReleaseCase();
                assistant.BargeInValidation -= OnBargeInValidation;
                await assistant.StopAsync().ConfigureAwait(false);
            }

            void OnBargeInValidation(object? sender, BargeInValidationEventArgs args)
            {
                if (args.Kind != BargeInValidationKind.Heard ||
                    args.Source != BargeInTriggerSource.NearEndWindow)
                {
                    return;
                }

                heard = args;
                detected = true;
            }
        }
        finally
        {
            // Also covers constructor/setup failures before the inner lifecycle finally exists.
            _tts.ReleaseCase();
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        try
        {
            _services.Dispose();
        }
        finally
        {
            _audio.Dispose();
            _mainAec.Dispose();
        }
    }

    private VoiceAssistantOptions CreateAssistantOptions()
        => new()
        {
            MinSentenceChars = 1,
            PlaybackTailBufferMs = 0,
            PlaybackPrerollMs = 0,
            BargeIn =
            {
                Enable = true,
                PlaybackStartupGuardMs = _options.PlaybackStartupGuardMs,
                ActiveNearEndWindowMs = _options.ActiveNearEndWindowMs,
                MinDurationMs = _options.MinDurationMs,
                MinRms = _options.MinRms,
            },
            Listening =
            {
                MinDurationMs = 0,
                MinRms = 0,
            },
        };

    private static async Task WaitUntilAsync(
        Func<bool> condition,
        CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(5));
        while (!condition())
            await Task.Delay(1, timeout.Token).ConfigureAwait(false);
    }

    private static async Task AllowQueuedConfirmationToReachControlLoopAsync(
        VoiceAssistantBenchmarkProbe probe,
        Func<bool> heardPublished,
        TimeSpan callbackPeriod,
        CancellationToken cancellationToken)
    {
        if (!ShouldWaitForQueuedConfirmation(probe, heardPublished()))
            return;

        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        // The reducer may finish during PushFrame while the single publication worker still
        // owes the Heard event. Two callback periods covers that trailing delivery without
        // advancing the virtual source clock.
        deadline.CancelAfter(callbackPeriod + callbackPeriod);
        try
        {
            while (!heardPublished())
            {
                if (!ShouldWaitForQueuedConfirmation(probe, heardPublished: false))
                    return;

                await Task.Delay(TimeSpan.FromMilliseconds(1), deadline.Token)
                    .ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException) when (
            !cancellationToken.IsCancellationRequested && deadline.IsCancellationRequested)
        {
            throw new TimeoutException(
                "The control loop did not publish a queued near-end confirmation within " +
                "two callback periods.");
        }
    }

    internal static bool ShouldWaitForQueuedConfirmation(
        VoiceAssistantBenchmarkProbe probe,
        bool heardPublished)
    {
        if (heardPublished)
            return false;

        VoiceAssistantBenchmarkSnapshot snapshot = probe.GetSnapshot();
        return !probe.InterruptionAdmissionsSettled ||
               snapshot.Coordinator.ConfirmationPending ||
               snapshot.Coordinator.ActiveTurnId != 0;
    }

    private static void CopyAvailable(float[] source, int offset, Span<float> destination)
    {
        if (offset >= source.Length)
            return;
        int count = Math.Min(destination.Length, source.Length - offset);
        source.AsSpan(offset, count).CopyTo(destination);
    }
}
