using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using VoiceBox.APM.WebRtc;
using VoiceBox.APM.WebRtc.Models;
using VoiceBox.ASR.Qwen3;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Adapters;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.TTS.VoxCPM;
using VoiceBox.VAD.FireRed;
using VoiceBox.VAD.FireRed.Models;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed class EndToEndEvaluator : IDisposable
{
    /// <summary>
    /// Committed device output latency modelled by the harness, in milliseconds. Mirrors the
    /// <c>SuggestedLatency = 0.020</c> used by every shipping VoiceAssistantDemo plan.
    /// </summary>
    private const int DeviceOutputLatencyMs = 20;

    private readonly EndToEndOptions _options;
    private readonly EvaluationDatasetManifest _manifest;
    private readonly VoxCPMTTSService _synthesizer;
    private readonly Dictionary<string, TtsResult> _ttsCache =
        new(StringComparer.Ordinal);
    private readonly BenchmarkTimeProvider _timeProvider = new();
    private readonly WebRtcApmService _mainAec;
    private readonly BenchmarkClockedAecService _clockedMainAec;
    private readonly BenchmarkDuplexAudioIngress _audio;
    private readonly BootstrapThenRealVadService _vad;
    private readonly AudioServiceManager _services;
    private readonly BootstrapThenRealAsrService _asr;
    private readonly PreparedTtsService _tts;
    private readonly EndToEndLlmService _llm;
    private bool _disposed;

    public EndToEndEvaluator(EndToEndOptions options)
    {
        _options = options;
        _manifest = EvaluationDatasetManifest.Load(options.DatasetRoot);

        _synthesizer = new VoxCPMTTSService(Options.Create(new VoxCPMTTSOptions
        {
            ModelPath = options.TtsModelPath,
            ExecutionProvider = options.TtsExecutionProvider,
            DefaultVoice = options.TtsVoice,
            DefaultSeed = 23,
            MaxLen = 220,
        }));

        // Force the output format before constructing the Assistant service graph.
        int outputSampleRate = _synthesizer.OutputSampleRate;
        int outputChannels = _synthesizer.OutputChannels;
        _tts = new PreparedTtsService(outputSampleRate, outputChannels);
        _mainAec = new WebRtcApmService(Options.Create(new WebRtcApmOptions
        {
            SampleRate = EvaluationOptions.TargetSampleRate,
            ChannelCount = 1,
            EnableHighPassFilter = true,
            EnableEchoCanceller = true,
            // Noise suppression matches the shipping configuration (WebRtcApmOptions defaults to
            // true, and every VoiceAssistantDemo plan enables it). Running the benchmark with it off
            // measured a capture chain the product never uses, and it did so on the exact signal
            // class that produces every false accept in this suite: noise-bearing echo-only cases.
            EnableNoiseSuppression = true,
            // AGC2 stays off, matching the shipping plans: its adaptive gain lifts the noise floor
            // and AEC residual during quiet passages and can form a feedback loop.
            EnableAgc2 = false,
        }));
        _audio = new BenchmarkDuplexAudioIngress(
            _mainAec,
            _timeProvider,
            EvaluationOptions.TargetSampleRate,
            channels: 1)
        {
            // Match the shipping duplex configuration: SuggestedLatency 0.020 in every
            // VoiceAssistantDemo plan. Without this the harness lets a queue clear silence the
            // loudspeaker instantly, which no device can do, and ducking would take effect on audio
            // the device had already accepted.
            DeviceOutputLatencySamples =
                EvaluationOptions.TargetSampleRate * DeviceOutputLatencyMs / 1000,
            // The real ClearQueue stops and restarts the duplex stream, so the barge-in utterance is
            // recorded across an AEC reset and a capture-timeline break.
            // Measured to be near-inert today (CER 65.43% with it on vs 65.31% off, identical empty
            // recognition) because the hard stop lands after the user has finished speaking. It
            // becomes material as soon as the stop moves earlier, so the model stays enabled.
            ModelStreamRestartOnClear = true,
        };
        _llm = new EndToEndLlmService();
        var vadOptions = new FireRedVadOptions
        {
            SampleRate = EvaluationOptions.TargetSampleRate,
            ChannelCount = 1,
            ExecutionProvider = options.ExecutionProvider,
            ThreadsNum = 1,
            SpeechThreshold = 0.9f,
        };
        if (!string.IsNullOrWhiteSpace(options.VadModelDirectory))
        {
            vadOptions.VADModelPath = Path.Combine(
                options.VadModelDirectory,
                "fireredvad_stream_vad_with_cache.onnx");
            vadOptions.CMVNPath = Path.Combine(options.VadModelDirectory, "cmvn.ark");
        }
        _vad = new BootstrapThenRealVadService(
            new FireRedVoiceActivityDetectService(Options.Create(vadOptions)));
        IASRService realAsr = new Qwen3ASRService(
            Options.Create(new Qwen3ASROptions
            {
                ModelPath = options.AsrModelPath,
                ExecutionProvider = options.ExecutionProvider,
                ForceLanguage = "Chinese",
                ChunkSizeSec = 2.0,
                MaxConcurrentSessions = 2,
                MaxConcurrentInferences = 1,
            }));
        _asr = new BootstrapThenRealAsrService(realAsr);
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

    public async Task<EndToEndCaseResult> EvaluateAsync(
        PreparedEndToEndCase testCase,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        TtsResult ttsResult = await GetTtsAsync(
            testCase.PromptText,
            cancellationToken).ConfigureAwait(false);
        _tts.BeginCase(ttsResult);
        _llm.SetResponse(testCase.PromptText);
        _asr.BeginCase();
        _timeProvider.Reset();

        PreparedSpeechAsset? speech = testCase.SpeechAssetId is null
            ? null
            : _manifest.SpeechAssets.Single(
                item => string.Equals(
                    item.Id,
                    testCase.SpeechAssetId,
                    StringComparison.Ordinal));
        WaveAudio? nearSpeech = speech is null
            ? null
            : WaveAudioLoader.LoadMono(
                EvaluationDatasetManifest.ResolvePath(_options.DatasetRoot, speech.AudioPath),
                EvaluationOptions.TargetSampleRate);
        PreparedRirAsset echoRir = _manifest.RirAssets.Single(
            item => item.Id == testCase.EchoRirId);
        PreparedRirAsset? nearRir = testCase.NearRirId is null
            ? null
            : _manifest.RirAssets.Single(item => item.Id == testCase.NearRirId);
        PreparedNoiseAsset? noise = testCase.NoiseAssetId is null
            ? null
            : _manifest.NoiseAssets.Single(item => item.Id == testCase.NoiseAssetId);

        float[] echoRirSamples = WaveAudioLoader.LoadMono(
            EvaluationDatasetManifest.ResolvePath(_options.DatasetRoot, echoRir.Path),
            EvaluationOptions.TargetSampleRate).Samples;
        float[]? nearRirSamples = nearRir is null
            ? null
            : WaveAudioLoader.LoadMono(
                EvaluationDatasetManifest.ResolvePath(_options.DatasetRoot, nearRir.Path),
                EvaluationOptions.TargetSampleRate).Samples;
        float[]? noiseSamples = noise is null
            ? null
            : WaveAudioLoader.LoadMono(
                EvaluationDatasetManifest.ResolvePath(_options.DatasetRoot, noise.Path),
                EvaluationOptions.TargetSampleRate).Samples;

        await using var _ = new AsyncCaseCleanup(_tts);
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
        using var assistant = builder.BuildForBenchmark(
            _timeProvider,
            probe,
            playbackBargeInDetectorFactory: null);

        var terminal = new TaskCompletionSource<BargeInValidationEventArgs>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        BargeInValidationEventArgs? heard = null;
        BargeInValidationEventArgs? timeoutAcceptance = null;
        string? recognizedFromUserSpoke = null;
        AssistantFaultedEventArgs? assistantFault = null;
        long heardSample = -1;
        bool asrTimedOut = false;
        try
        {
            assistant.BargeInValidation += OnValidation;
            assistant.UserSpoke += OnUserSpoke;
            assistant.Faulted += OnFaulted;
            await assistant.StartAsync(cancellationToken).ConfigureAwait(false);
            _vad.EmitBootstrapUtterance();
            await WaitUntilAsync(
                () => assistant.State == AssistantState.Speaking &&
                      _audio.IsPlaying &&
                      _audio.CapturedPlaybackSampleCount > 0,
                cancellationToken).ConfigureAwait(false);

            float[] capturedRender = await WaitForRenderAsync(
                ttsResult,
                cancellationToken).ConfigureAwait(false);
            // The scene resolves onset truth itself, so it needs the same eligibility rules the
            // product applies: nothing may be expected before the playback startup guard elapses or
            // before the minimum continuous-speech duration is available.
            AcousticSceneRenderer.RealtimeScene scene = AcousticSceneRenderer.CreateRealtime(
                testCase,
                capturedRender,
                nearSpeech?.Samples,
                echoRirSamples,
                nearRirSamples,
                noiseSamples,
                testCase.CallbackSamples,
                assistantOptions.BargeIn.PlaybackStartupGuardMs,
                assistantOptions.BargeIn.MinDurationMs,
                // Noise recorded in the case's own room already carries that room's acoustics;
                // convolving it again would double-reverberate it. Only geometry-less external
                // sources need spatializing.
                spatializeNoise: noise?.GeometryId is null);

            int callbackSamples = testCase.CallbackSamples;
            int quietEchoSamplesAfterClear = 0;
            for (int offset = 0; offset < scene.TotalSamples; offset += callbackSamples)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int length = Math.Min(callbackSamples, scene.TotalSamples - offset);
                var renderFrame = new float[callbackSamples];
                var microphoneFrame = new float[callbackSamples];
                var echoFrame = new float[callbackSamples];
                _audio.ReadPlayback(renderFrame);
                ApplyRenderGap(testCase, offset, renderFrame);
                scene.Process(
                    offset,
                    renderFrame,
                    microphoneFrame,
                    echoFrame);

                if (_audio.PlaybackClearCapturePosition >= 0)
                {
                    double echoRms = Rms(echoFrame.AsSpan(0, length));
                    quietEchoSamplesAfterClear = echoRms < 0.001
                        ? quietEchoSamplesAfterClear + length
                        : 0;
                    if (quietEchoSamplesAfterClear >=
                        EvaluationOptions.TargetSampleRate * 60 / 1000)
                    {
                        _audio.MarkPlaybackAudibleStop(offset + length);
                    }
                }

                TimeSpan callbackPeriod = TimeSpan.FromSeconds(
                    (double)length / EvaluationOptions.TargetSampleRate);
                _timeProvider.AdvanceSamples(length, EvaluationOptions.TargetSampleRate, 1);
                // A validation timeout can become due inside the clock advance. Give the single
                // non-audio consumer its budget before the capture cursor moves on.
                await AllowInterruptionDecisionToSettleAsync(
                    probe,
                    callbackPeriod,
                    cancellationToken).ConfigureAwait(false);
                // The event callbacks execute synchronously from PushFrame, so the local
                // sample cursor is an exact virtual-audio timestamp for Heard.
                _audio.PushFrame(
                    microphoneFrame.AsSpan(0, length),
                    renderFrame.AsSpan(0, length),
                    renderStatus: HasRenderGapOverlap(testCase, offset, length)
                        ? RenderTimelineStatus.Gap
                        : RenderTimelineStatus.Valid);
                await AllowInterruptionDecisionToSettleAsync(
                    probe,
                    callbackPeriod,
                    cancellationToken).ConfigureAwait(false);
                _audio.CommitPlaybackFrame();
                if (terminal.Task.IsCompleted &&
                    _audio.PlaybackClearCapturePosition >= 0 &&
                    _audio.PlaybackAudibleStopCapturePosition >= 0)
                {
                    break;
                }
                if ((offset / callbackSamples) % 16 == 0)
                    await Task.Yield();
            }

            if (heard is not null && !terminal.Task.IsCompleted)
            {
                using var timeout = new CancellationTokenSource(
                    TimeSpan.FromSeconds(_options.AsrTimeoutSeconds));
                try
                {
                    while (!terminal.Task.IsCompleted)
                    {
                        if (timeoutAcceptance is not null &&
                            (recognizedFromUserSpoke is not null ||
                             assistant.State == AssistantState.Idle))
                        {
                            // Empty/filler/echo ASR results after the stop do not emit a second
                            // validation event: the accepted stop is the only validation event
                            // and a meaningful whole-utterance result is exposed by UserSpoke.
                            terminal.TrySetResult(timeoutAcceptance);
                            break;
                        }

                        await Task.Delay(5, timeout.Token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (
                    !cancellationToken.IsCancellationRequested &&
                    timeout.IsCancellationRequested)
                {
                    // ASR may still be decoding a long/noisy clip after the stop. Keep the
                    // case scoreable instead of turning the harness wait budget into an
                    // execution error. If the assistant never accepted the interruption, the
                    // result remains an unaccepted Heard/ASR-timeout outcome.
                    asrTimedOut = true;
                    if (timeoutAcceptance is not null)
                        terminal.TrySetResult(timeoutAcceptance);
                }
            }

            BargeInValidationEventArgs? outcome =
                terminal.Task.IsCompletedSuccessfully ? terminal.Task.Result : null;
            BargeInValidationEventArgs? decision = heard ?? outcome;
            string recognized = recognizedFromUserSpoke ??
                                outcome?.Text ??
                                string.Empty;
            string reference = PreparedSpeechTranscript.ResolvePreparedReference(
                _manifest,
                testCase,
                speech);
            (int errors, int referenceCharacters, double? cer, bool exact) =
                ChineseTextMetrics.Compare(reference, recognized);
            // Latency reference.
            //
            // Reaction latency is measured from the earliest sample at which a correct
            // implementation is permitted to fire, which is the first point where enough continuous
            // near-end speech exists AND the playback startup guard has elapsed. The previous
            // reference was the sample at which the near-end clip was spliced into the timeline;
            // that charged the corpus recording's leading silence, the near-end RIR's propagation
            // delay and the product's own startup guard to the product as detection latency, and it
            // varied per clip in a way nothing in the report could explain.
            long latencyReference = scene.NearEndEligibleOnsetSample >= 0
                ? scene.NearEndEligibleOnsetSample
                : scene.NearEndSpeechOnsetSample;

            static double? LatencyMsFrom(long reference, long observed)
                => reference >= 0 && observed >= 0
                    ? (observed - reference) * 1000.0 / EvaluationOptions.TargetSampleRate
                    : null;

            // Firing before ANY near-end energy exists is a genuine false trigger regardless of the
            // eligibility rules, so this stays anchored to the acoustic onset.
            bool earlyFalseTrigger = testCase.OracleExpectHeard &&
                                     heard is not null &&
                                     heardSample >= 0 &&
                                     scene.NearEndSpeechOnsetSample >= 0 &&
                                     heardSample < scene.NearEndSpeechOnsetSample;
            double? heardLatency = testCase.OracleExpectHeard &&
                                   heard is not null &&
                                   heardSample >= 0
                ? LatencyMsFrom(latencyReference, heardSample)
                : null;
            double? heardLatencyFromSpeechOnset = testCase.OracleExpectHeard &&
                                                  heard is not null &&
                                                  heardSample >= 0
                ? LatencyMsFrom(scene.NearEndSpeechOnsetSample, heardSample)
                : null;
            double? duckLatency = testCase.OracleExpectDuck
                ? LatencyMsFrom(latencyReference, _audio.PlaybackDuckCapturePosition)
                : null;
            double? stopLatency = testCase.OracleExpectPlaybackStop
                ? LatencyMsFrom(latencyReference, _audio.PlaybackClearCapturePosition)
                : null;
            double? audibleStopLatency = testCase.OracleExpectPlaybackStop
                ? LatencyMsFrom(latencyReference, _audio.PlaybackAudibleStopCapturePosition)
                : null;
            // When the user actually stops hearing the assistant at full volume. Ducking is the only
            // arm of the barge-in sequence that runs in real time, so this is the number that
            // reflects perceived responsiveness.
            double? audibleDuckLatency = testCase.OracleExpectDuck
                ? LatencyMsFrom(latencyReference, _audio.PlaybackAudibleDuckCapturePosition)
                : null;
            // When the loudspeaker stopped emitting assistant audio entirely, excluding room
            // reverberation. Later than the control-plane stop by the committed device buffer.
            double? drainCompleteLatency = testCase.OracleExpectPlaybackStop
                ? LatencyMsFrom(latencyReference, _audio.PlaybackDrainCompleteCapturePosition)
                : null;
            // The part of the tail no implementation can remove: audio the device had already
            // accepted when the stop was issued.
            double? deviceTailLatency =
                testCase.OracleExpectPlaybackStop &&
                _audio.PlaybackDrainCompleteCapturePosition >= 0 &&
                _audio.PlaybackClearCapturePosition >= 0
                    ? (_audio.PlaybackDrainCompleteCapturePosition -
                       _audio.PlaybackClearCapturePosition) *
                      1000.0 / EvaluationOptions.TargetSampleRate
                    : null;
            double? audibleTailLatency =
                testCase.OracleExpectPlaybackStop &&
                _audio.PlaybackAudibleStopCapturePosition >= 0 &&
                _audio.PlaybackClearCapturePosition >= 0
                    ? (_audio.PlaybackAudibleStopCapturePosition -
                       _audio.PlaybackClearCapturePosition) *
                      1000.0 / EvaluationOptions.TargetSampleRate
                    : null;

            DumpControlTraceIfRequested(testCase, probe);
            return new EndToEndCaseResult
            {
                Id = testCase.Id,
                ClusterId = testCase.ClusterId,
                Corpus = speech?.Corpus ?? "echo-only",
                SpeechAssetId = testCase.SpeechAssetId,
                ExpectedNearEnd = testCase.ExpectedNearEnd,
                Scenario = testCase.Scenario,
                GeometryId = testCase.GeometryId,
                SpeakerPolicy = testCase.SpeakerPolicy,
                ExpectHeard = testCase.OracleExpectHeard,
                ExpectDuck = testCase.OracleExpectDuck,
                ExpectAccepted = testCase.OracleExpectAccepted,
                ExpectPlaybackStop = testCase.OracleExpectPlaybackStop,
                ExpectPlaybackResume = testCase.OracleExpectPlaybackResume,
                ReferenceText = reference,
                RecognizedText = recognized,
                Heard = heard is not null,
                EarlyFalseTrigger = earlyFalseTrigger,
                Accepted = outcome?.Kind == BargeInValidationKind.Accepted,
                Rejected = outcome?.Kind == BargeInValidationKind.Rejected,
                PlaybackDucked = _audio.PlaybackDuckCapturePosition >= 0,
                PlaybackStopped = _audio.PlaybackClearCapturePosition >= 0,
                PlaybackResumed = _audio.PlaybackResumeCapturePosition >= 0,
                ValidationReason = outcome?.Reason ??
                                  (asrTimedOut ? "asr-timeout" : null),
                CallbackSamples = callbackSamples,
                AudioDurationMs = scene.TotalSamples * 1000.0 /
                                  EvaluationOptions.TargetSampleRate,
                HeardLatencyMs = heardLatency,
                HeardLatencyFromSpeechOnsetMs = heardLatencyFromSpeechOnset,
                DuckLatencyMs = duckLatency,
                AudibleDuckLatencyMs = audibleDuckLatency,
                StopLatencyMs = stopLatency,
                AudibleStopLatencyMs = audibleStopLatency,
                AudibleTailLatencyMs = audibleTailLatency,
                DrainCompleteLatencyMs = drainCompleteLatency,
                DeviceTailLatencyMs = deviceTailLatency,
                NearEndInsertionOnsetMs = scene.NearEndOnsetSample * 1000.0 /
                                          EvaluationOptions.TargetSampleRate,
                NearEndSpeechOnsetMs = scene.NearEndSpeechOnsetSample >= 0
                    ? scene.NearEndSpeechOnsetSample * 1000.0 /
                      EvaluationOptions.TargetSampleRate
                    : null,
                NearEndEligibleOnsetMs = scene.NearEndEligibleOnsetSample >= 0
                    ? scene.NearEndEligibleOnsetSample * 1000.0 /
                      EvaluationOptions.TargetSampleRate
                    : null,
                NearEndEligible = scene.NearEndEligibleOnsetSample >= 0,
                NearEndActiveDurationMs = scene.NearEndActivity.ActiveDurationMs,
                TriggerSource = decision?.Source.ToString(),
                DecisionFrameCount = decision?.AudioFrameCount ?? 0,
                DecisionAcceptedFrameCount = decision?.AudioAcceptedFrameCount ?? 0,
                DecisionEchoLikeFrameCount = decision?.AudioEchoLikeFrameCount ?? 0,
                DecisionAcceptedDurationMs = decision?.AudioAcceptedDurationMs ?? 0,
                DecisionScoreDurationMs = decision?.AudioScoreDurationMs ?? 0,
                DecisionPeakScoreDurationMs = decision?.AudioPeakScoreDurationMs ?? 0,
                CharacterErrors = errors,
                ReferenceCharacters = referenceCharacters,
                CharacterErrorRate = cer,
                ExactMatch = exact,
                EmptyRecognition = string.IsNullOrWhiteSpace(recognized),
                SignalToEchoRatioDb = testCase.SignalToEchoRatioDb,
                SignalToNoiseRatioDb = testCase.SignalToNoiseRatioDb,
                // Realized levels are reported so the requested SER can be audited against what the
                // scene actually produced, instead of being taken on trust.
                NearEndActiveLevel = scene.NearEndActiveLevel,
                EchoActiveLevel = scene.EchoActiveLevel,
                RealizedSignalToEchoRatioDb =
                    scene.NearEndActiveLevel > 1e-9 && scene.EchoActiveLevel > 1e-9
                        ? 20 * Math.Log10(scene.NearEndActiveLevel / scene.EchoActiveLevel)
                        : null,
                EchoReturnLossDb = scene.EchoReturnLossDb,
                SpeakerGain = scene.SpeakerGain,
                ClippedSampleRatio = scene.ClippedSampleRatio,
                EchoDelayMs = testCase.EchoDelayMs,
                ClockDriftPpm = testCase.ClockDriftPpm,
                FarEndNonlinear = testCase.FarEndNonlinear,
                RenderGapStartMs = testCase.RenderGapStartMs,
                RenderGapDurationMs = testCase.RenderGapDurationMs,
                EchoRirId = testCase.EchoRirId,
                NearRirId = testCase.NearRirId,
                NoiseAssetId = testCase.NoiseAssetId,
                NoiseKind = noise?.Kind,
                NoiseCorpus = noise?.Corpus,
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            AssistantFaultedEventArgs? fatalFault = Volatile.Read(ref assistantFault);
            string error = fatalFault is null
                ? ex.ToString()
                : $"{fatalFault.Component}: {fatalFault.Exception}" +
                  $"{Environment.NewLine}Subsequent exception: {ex}";
            DumpControlTraceIfRequested(testCase, probe);
            return new EndToEndCaseResult
            {
                Id = testCase.Id,
                ClusterId = testCase.ClusterId,
                Corpus = speech?.Corpus ?? "echo-only",
                SpeechAssetId = testCase.SpeechAssetId,
                ExpectedNearEnd = testCase.ExpectedNearEnd,
                Scenario = testCase.Scenario,
                GeometryId = testCase.GeometryId,
                SpeakerPolicy = testCase.SpeakerPolicy,
                ExpectHeard = testCase.OracleExpectHeard,
                ExpectDuck = testCase.OracleExpectDuck,
                ExpectAccepted = testCase.OracleExpectAccepted,
                ExpectPlaybackStop = testCase.OracleExpectPlaybackStop,
                ExpectPlaybackResume = testCase.OracleExpectPlaybackResume,
                ReferenceText = PreparedSpeechTranscript.ResolvePreparedReference(
                    _manifest,
                    testCase,
                    speech),
                RecognizedText = string.Empty,
                Heard = heard is not null,
                EarlyFalseTrigger = false,
                Accepted = false,
                Rejected = false,
                PlaybackDucked = _audio.PlaybackDuckCapturePosition >= 0,
                PlaybackStopped = _audio.PlaybackClearCapturePosition >= 0,
                PlaybackResumed = _audio.PlaybackResumeCapturePosition >= 0,
                ValidationReason = "error",
                CallbackSamples = testCase.CallbackSamples,
                AudioDurationMs = 0,
                HeardLatencyMs = null,
                DuckLatencyMs = null,
                StopLatencyMs = null,
                AudibleStopLatencyMs = null,
                AudibleTailLatencyMs = null,
                TriggerSource = heard?.Source.ToString(),
                DecisionFrameCount = heard?.AudioFrameCount ?? 0,
                DecisionAcceptedFrameCount = heard?.AudioAcceptedFrameCount ?? 0,
                DecisionEchoLikeFrameCount = heard?.AudioEchoLikeFrameCount ?? 0,
                DecisionAcceptedDurationMs = heard?.AudioAcceptedDurationMs ?? 0,
                DecisionScoreDurationMs = heard?.AudioScoreDurationMs ?? 0,
                DecisionPeakScoreDurationMs = heard?.AudioPeakScoreDurationMs ?? 0,
                CharacterErrors = 0,
                ReferenceCharacters = 0,
                CharacterErrorRate = null,
                ExactMatch = false,
                EmptyRecognition = true,
                SignalToEchoRatioDb = testCase.SignalToEchoRatioDb,
                SignalToNoiseRatioDb = testCase.SignalToNoiseRatioDb,
                EchoDelayMs = testCase.EchoDelayMs,
                ClockDriftPpm = testCase.ClockDriftPpm,
                FarEndNonlinear = testCase.FarEndNonlinear,
                RenderGapStartMs = testCase.RenderGapStartMs,
                RenderGapDurationMs = testCase.RenderGapDurationMs,
                EchoRirId = testCase.EchoRirId,
                NearRirId = testCase.NearRirId,
                NoiseAssetId = testCase.NoiseAssetId,
                NoiseKind = noise?.Kind,
                NoiseCorpus = noise?.Corpus,
                Error = error,
            };
        }
        finally
        {
            assistant.BargeInValidation -= OnValidation;
            assistant.UserSpoke -= OnUserSpoke;
            assistant.Faulted -= OnFaulted;
            _tts.ReleaseCase();
            await assistant.StopAsync().ConfigureAwait(false);
        }

        void OnUserSpoke(object? sender, UserSpokeEventArgs args)
        {
            if (heard is not null)
                recognizedFromUserSpoke = args.Text;
        }

        void OnFaulted(object? sender, AssistantFaultedEventArgs args)
            => Interlocked.CompareExchange(ref assistantFault, args, null);

        void OnValidation(object? sender, BargeInValidationEventArgs args)
        {
            if (args.Kind == BargeInValidationKind.Heard && heard is null)
            {
                heard = args;
                heardSample = _audio.CurrentCaptureSamplePosition;
            }
            // Accepted is the stop decision (acoustic Interrupt with KWS off, speaker Match with
            // KWS on); it is issued before any recognition text exists. The whole-utterance
            // transcript follows through UserSpoke, or the turn ends without a response. Do not
            // let the stop event make the evaluator score an empty transcript.
            if (args.Kind == BargeInValidationKind.Accepted)
            {
                timeoutAcceptance ??= args;
                if (!testCase.OracleExpectAccepted)
                    terminal.TrySetResult(args);
            }
            else if (args.Kind != BargeInValidationKind.Heard)
            {
                terminal.TrySetResult(args);
            }
        }

        static void DumpControlTraceIfRequested(
            PreparedEndToEndCase testCase,
            VoiceAssistantBenchmarkProbe probe)
        {
            if (!string.Equals(
                    Environment.GetEnvironmentVariable("VOICEBOX_BENCHMARK_TRACE"),
                    "1",
                    StringComparison.Ordinal))
            {
                return;
            }

            foreach (ControlTraceEntry entry in probe.GetControlTrace())
            {
                Console.Error.WriteLine(
                    $"[TRACE] {testCase.Id} seq={entry.Sequence} stage={entry.Stage} " +
                    $"capture={entry.CapturePosition} turn={entry.TurnId} op={entry.OperationId} " +
                    $"detail={entry.Detail ?? "-"}");
            }
        }
    }

    public EndToEndSummary Aggregate(IEnumerable<EndToEndCaseResult> cases)
        => EndToEndMetrics.Aggregate(cases);

    private async Task<TtsResult> GetTtsAsync(
        string prompt,
        CancellationToken cancellationToken)
    {
        if (_ttsCache.TryGetValue(prompt, out TtsResult? cached))
            return cached;
        TtsResult result = await _synthesizer.GetAudioAsync(
            prompt,
            new TtsSynthesisOptions
            {
                Voice = _options.TtsVoice,
                Seed = 23,
                MaxNewFrames = 220,
            },
            cancellationToken).ConfigureAwait(false);
        _ttsCache[prompt] = result;
        return result;
    }

    private async Task<float[]> WaitForRenderAsync(
        TtsResult ttsResult,
        CancellationToken cancellationToken)
    {
        int expected = Math.Max(
            1,
            (int)Math.Round(
                ttsResult.Samples.Length /
                (double)ttsResult.Channels *
                EvaluationOptions.TargetSampleRate /
                ttsResult.SampleRate));
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(30));
        while (_audio.CapturedPlaybackSampleCount < Math.Max(320, expected - 512))
        {
            await Task.Delay(5, timeout.Token).ConfigureAwait(false);
        }
        return _audio.CopyCapturedPlayback();
    }

    private VoiceAssistantOptions CreateAssistantOptions()
    {
        var options = new VoiceAssistantOptions
        {
            TtsVoice = _options.TtsVoice,
        };
        options.BargeIn.Enable = true;
        return options;
    }

    private static void ApplyRenderGap(
        PreparedEndToEndCase testCase,
        int frameOffset,
        Span<float> render)
    {
        if (testCase.RenderGapStartMs is not int gapStartMs ||
            testCase.RenderGapDurationMs <= 0)
        {
            return;
        }
        int gapStart = checked(
            gapStartMs * EvaluationOptions.TargetSampleRate / 1000);
        int gapEnd = checked(
            gapStart +
            testCase.RenderGapDurationMs * EvaluationOptions.TargetSampleRate / 1000);
        int localStart = Math.Max(0, gapStart - frameOffset);
        int localEnd = Math.Min(render.Length, gapEnd - frameOffset);
        if (localEnd > localStart)
            render[localStart..localEnd].Clear();
    }

    private static bool HasRenderGapOverlap(
        PreparedEndToEndCase testCase,
        int frameOffset,
        int frameLength)
    {
        if (testCase.RenderGapStartMs is not int gapStartMs ||
            testCase.RenderGapDurationMs <= 0)
        {
            return false;
        }
        int gapStart = checked(
            gapStartMs * EvaluationOptions.TargetSampleRate / 1000);
        int gapEnd = checked(
            gapStart +
            testCase.RenderGapDurationMs * EvaluationOptions.TargetSampleRate / 1000);
        int frameEnd = checked(frameOffset + frameLength);
        return frameOffset < gapEnd && frameEnd > gapStart;
    }

    private static double Rms(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty)
            return 0;
        double energy = 0;
        foreach (float sample in samples)
            energy += sample * sample;
        return Math.Sqrt(energy / samples.Length);
    }

    private static async Task WaitUntilAsync(
        Func<bool> condition,
        CancellationToken cancellationToken)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(30));
        while (!condition())
            await Task.Delay(1, timeout.Token).ConfigureAwait(false);
    }

    /// <summary>
    /// Yields scheduling to the single non-audio control consumer while a real-time interruption
    /// decision is still admitted but unreduced. The dataset clock does not advance here and no
    /// PCM, DTD input, decision, threshold or gate changes; the wait only stops the harness from
    /// running virtual audio beyond the frozen callback-period budget. Exceeding two
    /// callback periods is an evaluation error instead of a silent clock advance.
    /// </summary>
    private static async Task AllowInterruptionDecisionToSettleAsync(
        VoiceAssistantBenchmarkProbe probe,
        TimeSpan callbackPeriod,
        CancellationToken cancellationToken)
    {
        if (probe.InterruptionAdmissionsSettled)
            return;

        TimeSpan budget = callbackPeriod + callbackPeriod;
        long started = Stopwatch.GetTimestamp();
        var spin = new SpinWait();
        while (!probe.InterruptionAdmissionsSettled)
        {
            if (Stopwatch.GetElapsedTime(started) > budget)
            {
                throw new TimeoutException(
                    "The control loop did not reduce an admitted interruption decision within " +
                    "two callback periods.");
            }

            if (spin.NextSpinWillYield)
                await Task.Delay(1, cancellationToken).ConfigureAwait(false);
            else
                spin.SpinOnce();
        }
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _services.Dispose();
        _vad.Dispose();
        _asr.Dispose();
        _synthesizer.Dispose();
        _audio.Dispose();
        _mainAec.Dispose();
    }

    private sealed class AsyncCaseCleanup : IAsyncDisposable
    {
        private readonly PreparedTtsService _tts;

        public AsyncCaseCleanup(PreparedTtsService tts) => _tts = tts;

        public ValueTask DisposeAsync()
        {
            _tts.ReleaseCase();
            return ValueTask.CompletedTask;
        }
    }
}
