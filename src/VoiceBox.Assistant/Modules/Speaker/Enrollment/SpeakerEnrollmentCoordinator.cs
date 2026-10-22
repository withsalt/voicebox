using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Exceptions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Enrollment;

/// <summary>
/// Owner enrollment on the identity tap (after the capture processor / AEC / APM, before any
/// independent DNR) — the same tap and fingerprint the online verifier uses. It records several
/// clean segments, optionally several controlled double-talk segments while a calibration stimulus
/// plays through the real output path, runs an independent confirmation per bank with the online
/// window rules, and only then overwrites the owner-speaker-profile-v2 file for the current front end.
/// The assistant must not be running; the duplex service is used exclusively.
/// </summary>
public sealed class SpeakerEnrollmentCoordinator
{
    private readonly IAudioDuplexService _audio;
    private readonly IVoiceActivityDetectService _vad;
    private readonly ISpeakerService _speaker;
    private readonly VoiceAssistantSpeakerOptions _speakerOptions;
    private readonly SpeakerEnrollmentOptions _options;
    private readonly ISpeakerEnrollmentStimulusPlayer? _stimulus;
    private readonly IPlaybackBargeInDetector? _nearEndDetector;
    private readonly ILogger _logger;
    private readonly SpeakerCaptureDescriptor _descriptor;
    private readonly SpeakerCaptureFrameFactory _frames;
    private readonly SpeakerEnrollmentSegmentCollector _collector = new();
    private readonly SpeakerEnrollmentQualityEvaluator _quality;
    private readonly SpeakerProfileBuilder _builder;
    private readonly float _threshold;
    private volatile bool _doubleTalkActive;
    private SpeakerEnrollmentProgressEventArgs? _activeSegment;
    private int _running;
    private EnrollmentCheckpoint? _pending;

    public SpeakerEnrollmentCoordinator(
        IAudioDuplexService audio,
        IVoiceActivityDetectService vad,
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions speakerOptions,
        SpeakerEnrollmentOptions? enrollmentOptions = null,
        ISpeakerEnrollmentStimulusPlayer? stimulusPlayer = null,
        IAcousticProcessingMetricsSource? acousticDiagnostics = null,
        ILogger<SpeakerEnrollmentCoordinator>? logger = null)
        : this(audio, vad, speaker, speakerOptions, enrollmentOptions, stimulusPlayer, acousticDiagnostics, nearEndDetector: null, logger)
    {
    }

    /// <summary>
    /// Internal overload that also runs the product's render-aware detector during double-talk
    /// segments so only frames with accepted near-end evidence count towards the segment.
    /// </summary>
    internal SpeakerEnrollmentCoordinator(
        IAudioDuplexService audio,
        IVoiceActivityDetectService vad,
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions speakerOptions,
        SpeakerEnrollmentOptions? enrollmentOptions,
        ISpeakerEnrollmentStimulusPlayer? stimulusPlayer,
        IAcousticProcessingMetricsSource? acousticDiagnostics,
        IPlaybackBargeInDetector? nearEndDetector,
        ILogger<SpeakerEnrollmentCoordinator>? logger)
    {
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _vad = vad ?? throw new ArgumentNullException(nameof(vad));
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _speakerOptions = speakerOptions ?? throw new ArgumentNullException(nameof(speakerOptions));
        _options = enrollmentOptions ?? new SpeakerEnrollmentOptions();
        _stimulus = stimulusPlayer;
        _nearEndDetector = nearEndDetector;
        _logger = logger ?? NullLogger<SpeakerEnrollmentCoordinator>.Instance;
        ValidateEnrollmentOptions(_options);
        _descriptor = SpeakerCaptureDescriptor.FromDuplexService(_audio, acousticDiagnostics);
        ValidateAudioFormat();
        _frames = new SpeakerCaptureFrameFactory(_descriptor);
        _quality = new SpeakerEnrollmentQualityEvaluator(_speaker, _options);
        _builder = new SpeakerProfileBuilder(_speaker, _options);
        _threshold = _speakerOptions.Threshold ?? _speaker.Threshold;
    }

    public event EventHandler<SpeakerEnrollmentProgressEventArgs>? ProgressChanged;

    public string ProfilePath => _speakerOptions.ResolveProfilePath();

    /// <summary>Frozen identity-tap fingerprint the new variant will carry.</summary>
    public string FrontendFingerprint => _descriptor.FrontendFingerprint;

    /// <summary>
    /// Accepted segments kept after a failed <see cref="EnrollAsync()"/>; the next call continues
    /// here instead of re-recording earlier segments. Null when there is nothing to resume.
    /// </summary>
    public SpeakerEnrollmentResumeInfo? PendingResume =>
        _pending is null
            ? null
            : new SpeakerEnrollmentResumeInfo(
                _pending.NextStage,
                _pending.NextIndex,
                _pending.Clean.Count,
                _pending.DoubleTalk.Count,
                _pending.CleanConfirmationPassed);

    /// <summary>Drops kept segments so the next <see cref="EnrollAsync()"/> starts from clean segment 1.</summary>
    public void DiscardPendingEnrollment()
    {
        if (Volatile.Read(ref _running) != 0)
            throw new InvalidOperationException("Cannot discard a pending enrollment while enrollment is running.");
        _pending = null;
    }

    /// <summary>True when a v2 profile exists and holds a usable variant for the current front end.</summary>
    public bool HasUsableProfile(bool requireDoubleTalkBank = false)
    {
        try
        {
            OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, ProfilePath);
            var banks = new List<SpeakerPurposeBank> { SpeakerPurposeBank.CleanNearEnd };
            if (requireDoubleTalkBank)
                banks.Add(SpeakerPurposeBank.DoubleTalkNearEnd);
            _ = SpeakerProfileStore.RequireVariant(profile, _descriptor, banks, ProfilePath);
            return true;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// Records every segment back-to-back and completes the profile. A later call on the same
    /// instance after a quality-gate or confirmation failure continues from the failed segment
    /// and keeps embeddings that already passed.
    /// </summary>
    public Task<SpeakerEnrollmentResult> EnrollAsync(CancellationToken ct = default)
        => EnrollCoreAsync(pacer: null, ct);

    /// <summary>
    /// Records the profile, asking <paramref name="pacer"/> before each segment attempt so an
    /// interactive host can show the prompt and wait for the user. Collection and the segment timeout
    /// only start after the pacer returns. A later call after a failure resumes from
    /// <see cref="PendingResume"/> unless <see cref="DiscardPendingEnrollment"/> was called.
    /// </summary>
    public Task<SpeakerEnrollmentResult> EnrollAsync(ISpeakerEnrollmentPacer pacer, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(pacer);
        return EnrollCoreAsync(pacer, ct);
    }

    private async Task<SpeakerEnrollmentResult> EnrollCoreAsync(ISpeakerEnrollmentPacer? pacer, CancellationToken ct)
    {
        if (Interlocked.CompareExchange(ref _running, 1, 0) != 0)
            throw new InvalidOperationException("Speaker enrollment is already running.");
        if (_audio.IsRecording || _audio.IsPlaying)
        {
            Interlocked.Exchange(ref _running, 0);
            throw new InvalidOperationException("Cannot enroll a speaker while the duplex audio service is already running.");
        }

        bool wantDoubleTalk = _options.DoubleTalkSegmentCount > 0;
        if (wantDoubleTalk && _stimulus is null)
        {
            Interlocked.Exchange(ref _running, 0);
            throw new InvalidOperationException(
                "Double-talk enrollment requires an ISpeakerEnrollmentStimulusPlayer; pass DoubleTalkSegmentCount = 0 to enroll the clean bank only.");
        }

        DuplexAudioFrameHandler onDuplex = OnDuplexFrame;
        EventHandler<VADEventArgs> onSpeaking = OnSpeaking;
        _audio.OnDuplexAudioFrame += onDuplex;
        _vad.OnSpeaking += onSpeaking;
        _collector.Progress += OnCollectorProgress;

        bool audioStarted = false;
        bool resuming = _pending is not null;
        EnrollmentCheckpoint session = (_pending ?? new EnrollmentCheckpoint()).Clone();
        try
        {
            RaiseProgress(new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.PreparingFrontend });
            if (resuming)
            {
                _logger.LogInformation(
                    "Resuming owner enrollment from {Stage} segment {Index} (kept clean={Clean}, doubleTalk={DoubleTalk}).",
                    session.NextStage,
                    session.NextIndex + 1,
                    session.Clean.Count,
                    session.DoubleTalk.Count);
                RaiseProgress(new SpeakerEnrollmentProgressEventArgs
                {
                    Kind = SpeakerEnrollmentProgressKind.Resumed,
                    Stage = session.NextStage,
                    SegmentIndex = session.NextIndex,
                    SegmentCount = session.NextStage switch
                    {
                        SpeakerEnrollmentStage.CleanEnrollment => _options.CleanSegmentCount,
                        SpeakerEnrollmentStage.DoubleTalkEnrollment => _options.DoubleTalkSegmentCount,
                        _ => 1,
                    },
                    Attempt = 1,
                });
            }

            _vad.Reset();
            _nearEndDetector?.ResetStream();
            await _audio.StartAsync(cancellationToken: ct).ConfigureAwait(false);
            audioStarted = true;

            int requiredSamples = SamplesFor(_options.SegmentSpeech);
            int confirmationSamples = SamplesFor(_options.ConfirmationSpeech);
            SpeakerEnrollmentStage resumeStage = session.NextStage;
            int resumeIndex = session.NextIndex;

            if (resumeStage == SpeakerEnrollmentStage.CleanEnrollment)
            {
                await CollectSegmentsAsync(
                    SpeakerEnrollmentStage.CleanEnrollment,
                    SpeakerEnrollmentCondition.Clean,
                    _options.CleanSegmentCount,
                    requiredSamples,
                    session.Clean,
                    resumeIndex,
                    session,
                    pacer,
                    ct).ConfigureAwait(false);
                bool needDoubleTalk = wantDoubleTalk && session.DoubleTalk.Count < _options.DoubleTalkSegmentCount;
                resumeStage = needDoubleTalk
                    ? SpeakerEnrollmentStage.DoubleTalkEnrollment
                    : SpeakerEnrollmentStage.CleanConfirmation;
                resumeIndex = needDoubleTalk ? session.DoubleTalk.Count : 0;
                Persist(session, resumeStage, resumeIndex);
            }

            if (wantDoubleTalk && resumeStage == SpeakerEnrollmentStage.DoubleTalkEnrollment)
            {
                await CollectSegmentsAsync(
                    SpeakerEnrollmentStage.DoubleTalkEnrollment,
                    SpeakerEnrollmentCondition.DoubleTalk,
                    _options.DoubleTalkSegmentCount,
                    requiredSamples,
                    session.DoubleTalk,
                    resumeIndex,
                    session,
                    pacer,
                    ct).ConfigureAwait(false);
                resumeStage = SpeakerEnrollmentStage.CleanConfirmation;
                resumeIndex = 0;
                Persist(session, SpeakerEnrollmentStage.CleanConfirmation, 0);
            }

            RaiseProgress(new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.BuildingProfile });
            SpeakerBankBuildResult? cleanBank = _builder.TryBuildBank(session.Clean, _options.MinCleanSegmentsRetained);
            if (cleanBank is null || !cleanBank.Bank.IsUsable || cleanBank.Retained.Count < _options.MinCleanSegmentsRetained)
            {
                session.Clean.Clear();
                session.CleanConfirmationPassed = false;
                session.CleanConfirmationScore = 0;
                Persist(session, SpeakerEnrollmentStage.CleanEnrollment, 0);
                throw Failure(
                    SpeakerEnrollmentReasonCode.NotEnoughSegments,
                    cleanBank is null
                        ? "clean 段无法构建 bank。"
                        : $"clean 段保留 {cleanBank.Retained.Count} 段，少于要求的 {_options.MinCleanSegmentsRetained} 段（剔除离群 {cleanBank.Discarded.Count} 段）。",
                    SpeakerEnrollmentStage.CleanEnrollment,
                    0);
            }

            SpeakerBankBuildResult? doubleTalkBank = null;
            if (wantDoubleTalk)
            {
                doubleTalkBank = _builder.TryBuildBank(session.DoubleTalk, _options.MinDoubleTalkSegmentsRetained);
                if (doubleTalkBank is null || !doubleTalkBank.Bank.IsUsable || doubleTalkBank.Retained.Count < _options.MinDoubleTalkSegmentsRetained)
                {
                    session.DoubleTalk.Clear();
                    session.CleanConfirmationPassed = false;
                    session.CleanConfirmationScore = 0;
                    Persist(session, SpeakerEnrollmentStage.DoubleTalkEnrollment, 0);
                    throw Failure(
                        SpeakerEnrollmentReasonCode.NotEnoughSegments,
                        doubleTalkBank is null
                            ? "double-talk 段无法构建 bank。"
                            : $"double-talk 段保留 {doubleTalkBank.Retained.Count} 段，少于要求的 {_options.MinDoubleTalkSegmentsRetained} 段。",
                        SpeakerEnrollmentStage.DoubleTalkEnrollment,
                        0);
                }
            }

            // Independent confirmation: never enters the prototypes, evaluated with the online rules.
            double cleanConfirmation;
            if (session.CleanConfirmationPassed)
            {
                cleanConfirmation = session.CleanConfirmationScore;
            }
            else
            {
                Persist(session, SpeakerEnrollmentStage.CleanConfirmation, 0);
                try
                {
                    cleanConfirmation = await ConfirmAsync(
                        SpeakerEnrollmentStage.CleanConfirmation,
                        SpeakerEnrollmentCondition.Clean,
                        cleanBank.Bank,
                        _speakerOptions.Confirmation,
                        confirmationSamples,
                        pacer,
                        ct).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is SpeakerEnrollmentException or OperationCanceledException)
                {
                    Persist(session, SpeakerEnrollmentStage.CleanConfirmation, 0);
                    throw;
                }

                session.CleanConfirmationPassed = true;
                session.CleanConfirmationScore = cleanConfirmation;
                Persist(
                    session,
                    wantDoubleTalk ? SpeakerEnrollmentStage.DoubleTalkConfirmation : SpeakerEnrollmentStage.SavingProfile,
                    0);
            }

            double doubleTalkConfirmation = double.NaN;
            if (doubleTalkBank is not null)
            {
                if (session.DoubleTalkConfirmationPassed)
                {
                    doubleTalkConfirmation = session.DoubleTalkConfirmationScore;
                }
                else
                {
                    Persist(session, SpeakerEnrollmentStage.DoubleTalkConfirmation, 0);
                    try
                    {
                        doubleTalkConfirmation = await ConfirmAsync(
                            SpeakerEnrollmentStage.DoubleTalkConfirmation,
                            SpeakerEnrollmentCondition.DoubleTalk,
                            doubleTalkBank.Bank,
                            _speakerOptions.Confirmation,
                            confirmationSamples,
                            pacer,
                            ct).ConfigureAwait(false);
                    }
                    catch (Exception ex) when (ex is SpeakerEnrollmentException or OperationCanceledException)
                    {
                        Persist(session, SpeakerEnrollmentStage.DoubleTalkConfirmation, 0);
                        throw;
                    }

                    session.DoubleTalkConfirmationPassed = true;
                    session.DoubleTalkConfirmationScore = doubleTalkConfirmation;
                }
            }

            Persist(session, SpeakerEnrollmentStage.SavingProfile, 0);
            await _audio.StopAsync(ct).ConfigureAwait(false);
            audioStarted = false;

            RaiseProgress(new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.SavingProfile });
            var banks = new Dictionary<SpeakerPurposeBank, SpeakerPrototypeBank>
            {
                [SpeakerPurposeBank.CleanNearEnd] = cleanBank.Bank,
            };
            if (doubleTalkBank is not null)
                banks[SpeakerPurposeBank.DoubleTalkNearEnd] = doubleTalkBank.Bank;

            double confirmationMin = doubleTalkBank is null
                ? cleanConfirmation
                : Math.Min(cleanConfirmation, doubleTalkConfirmation);
            var variant = new OwnerSpeakerProfileVariant(
                Guid.NewGuid().ToString("N"),
                _descriptor.FrontendFingerprint,
                _descriptor.TapId,
                DateTimeOffset.UtcNow,
                banks,
                new OwnerSpeakerProfileQuality(
                    cleanBank.Retained.Count + (doubleTalkBank?.Retained.Count ?? 0),
                    doubleTalkBank is null ? 1 : 2,
                    cleanBank.PairwiseP10,
                    cleanBank.PairwiseMedian,
                    confirmationMin,
                    Math.Max(cleanBank.WithinSpeakerSpread, doubleTalkBank?.WithinSpeakerSpread ?? 0)));

            var profile = new OwnerSpeakerProfile(
                OwnerSpeakerProfile.OwnerSpeakerId,
                DateTimeOffset.UtcNow,
                SpeakerProfileStore.DescribeProvider(_speaker),
                [variant]);
            try
            {
                await SpeakerProfileStore.SaveAsync(_speaker, ProfilePath, profile, ct).ConfigureAwait(false);
            }
            catch
            {
                Persist(session, SpeakerEnrollmentStage.SavingProfile, 0);
                throw;
            }

            _logger.LogInformation(
                "Owner speaker profile saved: path={Path}, fingerprint={Fingerprint}, cleanPrototypes={Clean}, doubleTalkPrototypes={DoubleTalk}, retried={Retried}, discarded={Discarded}, confirmationMin={ConfirmationMin:F4}.",
                ProfilePath,
                _descriptor.FrontendFingerprint,
                cleanBank.Bank.Prototypes.Count,
                doubleTalkBank?.Bank.Prototypes.Count ?? 0,
                session.RetriedSegments,
                cleanBank.Discarded.Count + (doubleTalkBank?.Discarded.Count ?? 0),
                confirmationMin);
            _pending = null;
            RaiseProgress(new SpeakerEnrollmentProgressEventArgs { Stage = SpeakerEnrollmentStage.Completed });

            return new SpeakerEnrollmentResult(
                ProfilePath,
                variant.VariantId,
                _descriptor.FrontendFingerprint,
                cleanBank.Bank.Prototypes.Count,
                doubleTalkBank?.Bank.Prototypes.Count ?? 0,
                cleanBank.Retained.Count,
                doubleTalkBank?.Retained.Count ?? 0,
                session.RetriedSegments,
                cleanBank.Discarded.Count + (doubleTalkBank?.Discarded.Count ?? 0),
                cleanBank.PairwiseMedian,
                cleanBank.PairwiseP10,
                confirmationMin,
                Math.Max(cleanBank.WithinSpeakerSpread, doubleTalkBank?.WithinSpeakerSpread ?? 0),
                _threshold);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            RaiseProgress(new SpeakerEnrollmentProgressEventArgs
            {
                Stage = SpeakerEnrollmentStage.Failed,
                ReasonCode = (ex as SpeakerEnrollmentException)?.Reason ?? SpeakerEnrollmentReasonCode.None,
            });
            throw;
        }
        finally
        {
            _collector.End();
            _collector.Progress -= OnCollectorProgress;
            await StopStimulusAsync().ConfigureAwait(false);
            if (audioStarted || _audio.IsRecording || _audio.IsPlaying)
            {
                try { await _audio.StopAsync().ConfigureAwait(false); }
                catch (Exception ex) { _logger.LogWarning(ex, "Stop duplex audio after speaker enrollment failed."); }
            }
            _audio.OnDuplexAudioFrame -= onDuplex;
            _vad.OnSpeaking -= onSpeaking;
            try { _vad.Reset(); } catch { }
            Interlocked.Exchange(ref _running, 0);
        }
    }

    private async Task CollectSegmentsAsync(
        SpeakerEnrollmentStage stage,
        SpeakerEnrollmentCondition condition,
        int count,
        int requiredSamples,
        List<SpeakerEnrollmentSegmentEmbedding> segments,
        int startIndex,
        EnrollmentCheckpoint session,
        ISpeakerEnrollmentPacer? pacer,
        CancellationToken ct)
    {
        for (int index = startIndex; index < count; index++)
        {
            Persist(session, stage, index);
            SpeakerEnrollmentSegmentEvaluation evaluation = default;
            for (int attempt = 1; attempt <= _options.MaxAttemptsPerSegment; attempt++)
            {
                evaluation = await RecordSegmentAsync(
                    stage,
                    condition,
                    index,
                    count,
                    attempt,
                    requiredSamples,
                    pacer,
                    ct).ConfigureAwait(false);
                if (evaluation.Accepted)
                    break;
                session.RetriedSegments++;
                Persist(session, stage, index);
                _logger.LogInformation(
                    "Enrollment segment retried: condition={Condition}, segment={Segment}/{Count}, attempt={Attempt}, reason={Reason}.",
                    condition,
                    index + 1,
                    count,
                    attempt,
                    evaluation.Reason);
            }

            if (!evaluation.Accepted)
            {
                throw Failure(
                    SpeakerEnrollmentReasonCode.TooManyRetries,
                    $"{condition} 第 {index + 1} 段连续 {_options.MaxAttemptsPerSegment} 次未通过质量门（最后原因 {evaluation.Reason}）。",
                    stage,
                    index);
            }

            segments.Add(evaluation.Embedding!);
        }
    }

    private async Task<SpeakerEnrollmentSegmentEvaluation> RecordSegmentAsync(
        SpeakerEnrollmentStage stage,
        SpeakerEnrollmentCondition condition,
        int index,
        int count,
        int attempt,
        int requiredSamples,
        ISpeakerEnrollmentPacer? pacer,
        CancellationToken ct)
    {
        var announce = new SpeakerEnrollmentProgressEventArgs
        {
            Kind = SpeakerEnrollmentProgressKind.SegmentStarted,
            Stage = stage,
            Condition = condition,
            SegmentIndex = index,
            SegmentCount = count,
            Attempt = attempt,
            Collected = TimeSpan.Zero,
            Required = TimeSpan.FromSeconds(requiredSamples / (double)_speaker.SampleRate),
        };
        if (pacer is not null)
            await pacer.WaitForSegmentStartAsync(announce, ct).ConfigureAwait(false);
        Volatile.Write(ref _activeSegment, announce);
        RaiseProgress(announce);

        SpeakerEnrollmentSegmentCapture capture;
        try
        {
            capture = await CaptureAsync(condition, requiredSamples, ct).ConfigureAwait(false);
        }
        catch (SpeakerEnrollmentSegmentException ex)
        {
            RaiseSegmentOutcome(stage, condition, index, count, attempt, null, ex.Reason);
            return new SpeakerEnrollmentSegmentEvaluation(ex.Reason, SpeakerEnrollmentSegmentQuality.Empty, null);
        }
        catch (TimeoutException)
        {
            RaiseSegmentOutcome(stage, condition, index, count, attempt, null, SpeakerEnrollmentReasonCode.SegmentTimeout);
            return new SpeakerEnrollmentSegmentEvaluation(SpeakerEnrollmentReasonCode.SegmentTimeout, SpeakerEnrollmentSegmentQuality.Empty, null);
        }

        SpeakerEnrollmentSegmentEvaluation evaluation = _quality.Evaluate(
            capture,
            condition,
            index,
            requiredSamples,
            nearEndGateAvailable: _nearEndDetector is not null);
        RaiseSegmentOutcome(stage, condition, index, count, attempt, evaluation.Quality, evaluation.Reason);
        return evaluation;
    }

    private async Task<SpeakerEnrollmentSegmentCapture> CaptureAsync(
        SpeakerEnrollmentCondition condition,
        int requiredSamples,
        CancellationToken ct)
    {
        _vad.Reset();
        bool doubleTalk = condition == SpeakerEnrollmentCondition.DoubleTalk;
        if (doubleTalk)
        {
            _nearEndDetector?.ResetDecisionState();
            await _stimulus!.StartAsync(ct).ConfigureAwait(false);
        }
        _doubleTalkActive = doubleTalk;
        try
        {
            Task<SpeakerEnrollmentSegmentCapture> segment = _collector.Begin(requiredSamples);
            return await segment.WaitAsync(_options.SegmentTimeout, ct).ConfigureAwait(false);
        }
        finally
        {
            _doubleTalkActive = false;
            _collector.End();
            if (doubleTalk)
                await StopStimulusAsync().ConfigureAwait(false);
        }
    }

    private async Task<double> ConfirmAsync(
        SpeakerEnrollmentStage stage,
        SpeakerEnrollmentCondition condition,
        SpeakerPrototypeBank bank,
        SpeakerConfirmationOptions purpose,
        int requiredSamples,
        ISpeakerEnrollmentPacer? pacer,
        CancellationToken ct)
    {
        float threshold = purpose.Threshold ?? _threshold;
        SpeakerEnrollmentSegmentEvaluation evaluation = default;
        for (int attempt = 1; attempt <= _options.MaxAttemptsPerSegment; attempt++)
        {
            evaluation = await RecordSegmentAsync(
                stage,
                condition,
                0,
                1,
                attempt,
                requiredSamples,
                pacer,
                ct).ConfigureAwait(false);
            if (evaluation.Accepted)
                break;
        }
        if (!evaluation.Accepted)
        {
            throw Failure(
                SpeakerEnrollmentReasonCode.TooManyRetries,
                $"{condition} 确认段未通过质量门（{evaluation.Reason}）。",
                stage,
                0);
        }

        // Same window rule as online verification: every accepted window is scored against the bank;
        // the confirmation passes only when enough windows match by the purpose's normal rule.
        float[][] windows = evaluation.Embedding!.WindowEmbeddings;
        int matched = 0;
        float minScore = float.PositiveInfinity;
        foreach (float[] window in windows)
        {
            SpeakerWindowScore score = SpeakerProfileMatcher.Score(_speaker, bank, window);
            minScore = Math.Min(minScore, score.Reported);
            if (SpeakerProfileMatcher.IsMatch(in score, threshold))
                matched++;
        }

        int requiredMatches = Math.Max(1, (int)Math.Ceiling(windows.Length * purpose.RequiredMatches / (double)Math.Max(1, purpose.WindowCount)));
        bool passed = windows.Length > 0 && matched >= Math.Min(windows.Length, requiredMatches);
        _logger.LogInformation(
            "Enrollment confirmation: condition={Condition}, windows={Windows}, matched={Matched}, required={Required}, minScore={MinScore:F4}, threshold={Threshold:F4}, passed={Passed}.",
            condition,
            windows.Length,
            matched,
            requiredMatches,
            minScore,
            threshold,
            passed);
        if (!passed)
        {
            throw Failure(
                SpeakerEnrollmentReasonCode.ConfirmationFailed,
                $"{condition} 独立确认未通过：matched={matched}/{windows.Length}, minScore={minScore:F4}, threshold={threshold:F4}。将重试该确认段。",
                stage,
                0);
        }

        return float.IsFinite(minScore) ? minScore : 0;
    }

    private void OnDuplexFrame(in DuplexAudioFrame frame)
    {
        if (!_collector.IsActive)
            return;
        if (!_frames.TryCreate(in frame, out SpeakerCaptureFrame identity))
            return;

        bool renderActive = IsRenderActive(frame.RenderReference.Span);
        bool? nearEnd = null;
        if (_doubleTalkActive && _nearEndDetector is not null)
        {
            try
            {
                DtdProcessBatch batch = _nearEndDetector.Push(in frame, null, decisionEnabled: true, echoLearningEnabled: false);
                if (batch.HasObservation)
                    nearEnd = batch.IsNearEndSpeech;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Enrollment near-end detector push failed; near-end gate skipped for this frame.");
            }
        }

        try
        {
            _collector.OnIdentityFrame(in identity, renderActive, nearEnd);
            _vad.Predict(identity.Audio.Span);
        }
        catch (Exception ex)
        {
            _collector.Fail(ex);
        }
    }

    private void OnSpeaking(object? sender, VADEventArgs e)
    {
        if (e.Data.Length == 0)
            return;
        _collector.OnSpeech(e.Data);
    }

    private void OnCollectorProgress(int collected, int required)
    {
        SpeakerEnrollmentProgressEventArgs? segment = Volatile.Read(ref _activeSegment);
        RaiseProgress(new SpeakerEnrollmentProgressEventArgs
        {
            Kind = SpeakerEnrollmentProgressKind.Collecting,
            Stage = segment?.Stage ?? (_doubleTalkActive ? SpeakerEnrollmentStage.DoubleTalkEnrollment : SpeakerEnrollmentStage.CleanEnrollment),
            Condition = segment?.Condition ?? (_doubleTalkActive ? SpeakerEnrollmentCondition.DoubleTalk : SpeakerEnrollmentCondition.Clean),
            SegmentIndex = segment?.SegmentIndex ?? 0,
            SegmentCount = segment?.SegmentCount ?? 0,
            Attempt = segment?.Attempt ?? 0,
            Collected = TimeSpan.FromSeconds(collected / (double)_speaker.SampleRate),
            Required = TimeSpan.FromSeconds(required / (double)_speaker.SampleRate),
        });
    }

    private void RaiseSegmentOutcome(
        SpeakerEnrollmentStage stage,
        SpeakerEnrollmentCondition condition,
        int index,
        int count,
        int attempt,
        SpeakerEnrollmentSegmentQuality? quality,
        SpeakerEnrollmentReasonCode reason)
        => RaiseProgress(new SpeakerEnrollmentProgressEventArgs
        {
            Kind = SpeakerEnrollmentProgressKind.SegmentOutcome,
            Stage = stage,
            Condition = condition,
            SegmentIndex = index,
            SegmentCount = count,
            Attempt = attempt,
            Collected = quality?.Duration ?? TimeSpan.Zero,
            Required = _options.SegmentSpeech,
            QualitySnapshot = quality,
            ReasonCode = reason,
        });

    private bool IsRenderActive(ReadOnlySpan<float> render)
    {
        if (render.IsEmpty)
            return false;
        double energy = 0;
        for (int i = 0; i < render.Length; i++)
            energy += (double)render[i] * render[i];
        return Math.Sqrt(energy / render.Length) >= _options.RenderActiveRms;
    }

    private async Task StopStimulusAsync()
    {
        if (_stimulus is null)
            return;
        try { await _stimulus.StopAsync(CancellationToken.None).ConfigureAwait(false); }
        catch (Exception ex) { _logger.LogDebug(ex, "Stopping the enrollment stimulus failed."); }
    }

    private void RaiseProgress(SpeakerEnrollmentProgressEventArgs args)
    {
        foreach (Delegate subscriber in ProgressChanged?.GetInvocationList() ?? [])
        {
            try { ((EventHandler<SpeakerEnrollmentProgressEventArgs>)subscriber).Invoke(this, args); }
            catch (Exception ex) { _logger.LogDebug(ex, "Speaker enrollment progress subscriber failed."); }
        }
    }

    private int SamplesFor(TimeSpan duration)
        => checked((int)Math.Ceiling(duration.TotalSeconds * _speaker.SampleRate));

    private void ValidateAudioFormat()
    {
        if (_descriptor.SampleRate != _speaker.SampleRate || _descriptor.Channels != 1)
        {
            throw new InvalidOperationException(
                "Speaker enrollment requires the duplex identity tap to match the SpeakerService: " +
                $"audio={_descriptor.SampleRate}Hz/{_descriptor.Channels}ch, speaker={_speaker.SampleRate}Hz/1ch.");
        }
    }

    private static void ValidateEnrollmentOptions(SpeakerEnrollmentOptions options)
    {
        if (options.CleanSegmentCount < 1)
            throw new ArgumentException("CleanSegmentCount must be >= 1.", nameof(options));
        if (options.DoubleTalkSegmentCount < 0)
            throw new ArgumentException("DoubleTalkSegmentCount must be >= 0.", nameof(options));
        if (options.SegmentSpeech <= TimeSpan.Zero)
            throw new ArgumentException("SegmentSpeech must be > 0.", nameof(options));
        if (options.SegmentTimeout < options.SegmentSpeech)
            throw new ArgumentException("SegmentTimeout must be >= SegmentSpeech.", nameof(options));
        if (options.ConfirmationSpeech <= TimeSpan.Zero)
            throw new ArgumentException("ConfirmationSpeech must be > 0.", nameof(options));
        if (options.MaxAttemptsPerSegment < 1)
            throw new ArgumentException("MaxAttemptsPerSegment must be >= 1.", nameof(options));
        if (options.MinCleanSegmentsRetained < SpeakerPrototypeBank.MinPrototypes ||
            options.MinCleanSegmentsRetained > options.CleanSegmentCount)
        {
            throw new ArgumentException(
                $"MinCleanSegmentsRetained must be within [{SpeakerPrototypeBank.MinPrototypes}, CleanSegmentCount].",
                nameof(options));
        }
        if (options.DoubleTalkSegmentCount > 0 &&
            (options.MinDoubleTalkSegmentsRetained < SpeakerPrototypeBank.MinPrototypes ||
             options.MinDoubleTalkSegmentsRetained > options.DoubleTalkSegmentCount))
        {
            throw new ArgumentException(
                $"MinDoubleTalkSegmentsRetained must be within [{SpeakerPrototypeBank.MinPrototypes}, DoubleTalkSegmentCount].",
                nameof(options));
        }
        if (options.MaxPrototypesPerBank < options.MinPrototypesPerBank ||
            options.MinPrototypesPerBank < SpeakerPrototypeBank.MinPrototypes ||
            options.MaxPrototypesPerBank > SpeakerPrototypeBank.MaxPrototypes)
        {
            throw new ArgumentException("Prototype counts must satisfy 2 <= Min <= Max <= 8.", nameof(options));
        }
        if (!float.IsFinite(options.MinimumRms) || options.MinimumRms < 0f)
            throw new ArgumentException("MinimumRms must be finite and >= 0.", nameof(options));
    }

    private void Persist(EnrollmentCheckpoint session, SpeakerEnrollmentStage stage, int index)
    {
        session.NextStage = stage;
        session.NextIndex = index;
        _pending = session.Clone();
    }

    private static SpeakerEnrollmentException Failure(
        SpeakerEnrollmentReasonCode reason,
        string message,
        SpeakerEnrollmentStage stage = SpeakerEnrollmentStage.Failed,
        int segmentIndex = 0)
        => new(reason, message, stage, segmentIndex);

    private sealed class EnrollmentCheckpoint
    {
        public List<SpeakerEnrollmentSegmentEmbedding> Clean { get; } = [];
        public List<SpeakerEnrollmentSegmentEmbedding> DoubleTalk { get; } = [];
        public SpeakerEnrollmentStage NextStage { get; set; } = SpeakerEnrollmentStage.CleanEnrollment;
        public int NextIndex { get; set; }
        public int RetriedSegments { get; set; }
        public bool CleanConfirmationPassed { get; set; }
        public double CleanConfirmationScore { get; set; }
        public bool DoubleTalkConfirmationPassed { get; set; }
        public double DoubleTalkConfirmationScore { get; set; }

        public EnrollmentCheckpoint Clone()
        {
            var copy = new EnrollmentCheckpoint
            {
                NextStage = NextStage,
                NextIndex = NextIndex,
                RetriedSegments = RetriedSegments,
                CleanConfirmationPassed = CleanConfirmationPassed,
                CleanConfirmationScore = CleanConfirmationScore,
                DoubleTalkConfirmationPassed = DoubleTalkConfirmationPassed,
                DoubleTalkConfirmationScore = DoubleTalkConfirmationScore,
            };
            copy.Clean.AddRange(Clean);
            copy.DoubleTalk.AddRange(DoubleTalk);
            return copy;
        }
    }
}
