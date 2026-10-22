using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Diagnostics.Metrics.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Runtime.Pipelines;

/// <summary>
/// Render-reference data-plane path. Registered only when a playback barge-in detector is
/// installed; otherwise the router has no render sink.
/// </summary>
internal sealed class RenderReferencePipeline : IRenderReferenceFrameSink
{
    private readonly IPlaybackBargeInDetector _detector;
    private readonly AecMetricsSampler _aecMetrics;
    private readonly IPlaybackStartupGuardView _playbackStartupGuard;
    private readonly VoiceAssistantBargeInOptions _bargeInOptions;
    private readonly TimeProvider _timeProvider;
    private readonly BargeInTurnCoordinator _turns;
    private readonly INearEndRenderPort _nearEnd;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;
    private readonly ILogger _logger;
    private readonly IRuntimeLivenessView _runtime;
    private readonly IPlaybackActivityView _playback;
    private readonly IRenderAwareFaultSink _faults;
    private long _lastNearEndDiagnosticTicks;

    public RenderReferencePipeline(
        IPlaybackBargeInDetector detector,
        AecMetricsSampler aecMetrics,
        IPlaybackStartupGuardView playbackStartupGuard,
        VoiceAssistantBargeInOptions bargeInOptions,
        TimeProvider timeProvider,
        BargeInTurnCoordinator turns,
        INearEndRenderPort nearEnd,
        VoiceAssistantBenchmarkProbe? benchmarkProbe,
        ILogger logger,
        IRuntimeLivenessView runtime,
        IPlaybackActivityView playback,
        IRenderAwareFaultSink faults)
    {
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
        _aecMetrics = aecMetrics ?? throw new ArgumentNullException(nameof(aecMetrics));
        _playbackStartupGuard = playbackStartupGuard ?? throw new ArgumentNullException(nameof(playbackStartupGuard));
        _bargeInOptions = bargeInOptions ?? throw new ArgumentNullException(nameof(bargeInOptions));
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
        _turns = turns ?? throw new ArgumentNullException(nameof(turns));
        _nearEnd = nearEnd ?? throw new ArgumentNullException(nameof(nearEnd));
        _benchmarkProbe = benchmarkProbe;
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _faults = faults ?? throw new ArgumentNullException(nameof(faults));
    }

    public void OnRenderReference(in DuplexAudioFrame frame)
    {
        if (_runtime.IsStopping)
            return;

        DtdProcessBatch batch;
        RenderAwareBargeInSnapshot snapshot;
        BargeInDecision decision;
        bool startupGuardActive;
        bool playbackTransportActive = false;
        bool decisionEnabled = false;
        bool echoLearningEnabled = false;
        ActiveNearEndProcessingState processingState = default;
        try
        {
            if (frame.HasDiscontinuity)
                _aecMetrics.Reset();
            _aecMetrics.SampleIfDue(frame.TimestampTicks);

            RenderDelayHint? delayHint = null;
            if (_aecMetrics.TryGetLatest(out AcousticProcessingMetricsSnapshot metrics) &&
                IsFreshDelayHint(metrics.TimestampTicks, frame.TimestampTicks))
            {
                if (metrics.HasDelayMedianMs && metrics.DelayMedianMs >= 0)
                {
                    delayHint = new RenderDelayHint(
                        metrics.DelayMedianMs,
                        metrics.TimestampTicks > 0 ? metrics.TimestampTicks : frame.TimestampTicks,
                        IsStable: true);
                }
                else if (metrics.HasDelayMs && metrics.DelayMs >= 0)
                {
                    delayHint = new RenderDelayHint(
                        metrics.DelayMs,
                        metrics.TimestampTicks > 0 ? metrics.TimestampTicks : frame.TimestampTicks,
                        IsStable: false);
                }
            }

            startupGuardActive = _playbackStartupGuard.IsActiveAt(frame.TimestampTicks);
            playbackTransportActive = _playback.GetActivitySnapshot().IsActive;
            processingState = _nearEnd.GetProcessingState();
            // New candidates need an active reply to interrupt. An active capture, however, keeps
            // its decisions running after the owner-scoped stop has already silenced the reply: the
            // physical stop precedes the endpoint now, and the utterance still has to be followed
            // to its end on the same acoustic authority.
            decisionEnabled =
                !startupGuardActive &&
                ((playbackTransportActive && _turns.IsSpeakingWithoutPendingCandidate) ||
                 processingState.HasActiveCapture);
            echoLearningEnabled =
                playbackTransportActive &&
                !processingState.HasActiveCapture &&
                !processingState.HasPendingCandidate;
            batch = _detector.Push(
                in frame,
                delayHint,
                decisionEnabled,
                echoLearningEnabled);
            snapshot = _detector.GetSnapshot();
            decision = _detector.GetRecentDecision(
                TimeSpan.FromMilliseconds(_bargeInOptions.ActiveNearEndWindowMs),
                _bargeInOptions.MinDurationMs,
                _bargeInOptions.MinRms);
        }
        catch (ObjectDisposedException ex) when (
            _runtime.IsStopping)
        {
            _logger.LogDebug(ex, "Ignoring render-aware DTD callback during shutdown.");
            return;
        }
        catch (Exception ex)
        {
            _faults.ReportFatalDetectorFault(ex);
            return;
        }

        _nearEnd.ObserveDuplexAudio(
            in frame,
            startupGuardActive,
            batch.IsNearEndSpeech);
        // The diagnostic snapshot takes the near-end coordinator lock. Both consumers read
        // the same instant, so take it once, and skip it entirely when neither debug
        // logging nor a benchmark probe will read it.
        bool debugEnabled = _logger.IsEnabled(LogLevel.Debug);
        ActiveNearEndDiagnosticSnapshot nearEndDiagnostic =
            debugEnabled || _benchmarkProbe is not null
                ? _nearEnd.GetDiagnosticSnapshot()
                : default;
        if (debugEnabled)
        {
            LogNearEndDecisionDiagnosticsIfDue(
                batch,
                snapshot,
                decision,
                startupGuardActive,
                playbackTransportActive,
                decisionEnabled,
                echoLearningEnabled,
                processingState,
                nearEndDiagnostic);
        }
        _benchmarkProbe?.Observe(
            decision,
            snapshot,
            startupGuardActive,
            _playback.GetActivitySnapshot().IsActive,
            nearEndDiagnostic,
            _turns.ConversationState,
            _turns.IsSpeakingWithoutPendingCandidate);
    }

    private bool IsFreshDelayHint(long hintTimestampTicks, long frameTimestampTicks)
    {
        if (hintTimestampTicks <= 0 || frameTimestampTicks <= 0)
            return false;

        TimeSpan age = _timeProvider.GetElapsedTime(hintTimestampTicks, frameTimestampTicks);
        return age >= TimeSpan.Zero &&
            age <= TimeSpan.FromMilliseconds(_bargeInOptions.RenderAlignment.DelayHintMaxAgeMs);
    }

    private void LogNearEndDecisionDiagnosticsIfDue(
        DtdProcessBatch batch,
        RenderAwareBargeInSnapshot snapshot,
        BargeInDecision decision,
        bool startupGuardActive,
        bool playbackTransportActive,
        bool decisionEnabled,
        bool echoLearningEnabled,
        ActiveNearEndProcessingState processingState,
        ActiveNearEndDiagnosticSnapshot nearEndDiagnostic)
    {
        if (!_logger.IsEnabled(LogLevel.Debug) ||
            (_turns.ConversationState != AssistantState.Speaking &&
             decision.Action != BargeInAction.Interrupt &&
             !nearEndDiagnostic.CandidateActive &&
             nearEndDiagnostic.ActiveTurnId == 0))
        {
            return;
        }

        long now = Environment.TickCount64;
        long last = Volatile.Read(ref _lastNearEndDiagnosticTicks);
        if (now - last < 1000 ||
            Interlocked.CompareExchange(ref _lastNearEndDiagnosticTicks, now, last) != last)
        {
            return;
        }

        _logger.LogDebug(
            "Render-aware DTD: state={State}, decisionEnabled={DecisionEnabled}, " +
            "echoLearning={EchoLearningEnabled}, playback={PlaybackTransportActive}, " +
            "startupGuard={StartupGuardActive}, hops={HopCount}, observations={ObservationCount}, " +
            "probability={Probability:F3}/{SmoothedProbability:F3}, nearEnd={NearEnd}, " +
            "decision={Action}, reason={Reason}, frames={Frames}, acceptedFrames={AcceptedFrames}, " +
            "rejectedFrames={RejectedFrames}, echoFrames={EchoFrames}, totalMs={TotalMs:F0}, " +
            "acceptedMs={AcceptedMs:F0}, scoreMs={ScoreMs:F0}/{RequiredMs:F0}, peakScoreMs={PeakScoreMs:F0}, " +
            "latestAccepted={LatestAccepted}, rms={LastRms:F4}/{LastRequiredRms:F4}, " +
            "postAecFrames={PostAecFrames}, residual={MeanResidual:F3}, " +
            "corr={MeanCorrelation:F3}/{MeanPostAecCorrelation:F3}, discontinuity={Discontinuity}, " +
            "alignment={AlignmentReady}, delayMs={DelayMs:F2}, driftPpm={DriftPpm:F1}, " +
            "confidence={Confidence:F2}, generation={Generation}, renderStatus={RenderStatus}, " +
            "streamEpoch={StreamEpoch}, range=[{StartSample},{EndSample}), reset={StateReset}, " +
            "candidate={CandidateActive}/{CandidateAuthorized}, candidateOutcome={CandidateOutcome}, " +
            "candidateEnd={CandidateEndSample}, mainEnd={MainAudioEndSample}, activeTurn={ActiveTurnId}, " +
            "processing={HasActiveCapture}/{HasPendingCandidate}, " +
            "pathGuard={PathGuardActive}/{PathGuardRemainingMs:F0}ms, pathChanges={PathChangeCount}, " +
            "lastPathChange={LastPathChange}/{LastPathGuarded}/{LastPathChangeReason}.",
            _turns.ConversationState,
            decisionEnabled,
            echoLearningEnabled,
            playbackTransportActive,
            startupGuardActive,
            batch.HopCount,
            batch.ObservationCount,
            snapshot.LatestProbability,
            snapshot.LatestSmoothedProbability,
            snapshot.LatestNearEndSpeech,
            decision.Action,
            decision.Reason,
            decision.FrameCount,
            decision.AcceptedFrameCount,
            decision.RejectedFrameCount,
            decision.EchoLikeFrameCount,
            decision.TotalDurationMs,
            decision.AcceptedDurationMs,
            decision.ScoreDurationMs,
            decision.RequiredDurationMs,
            decision.PeakScoreDurationMs,
            decision.LatestFrameAccepted,
            decision.LastFrameRms,
            decision.LastRequiredRms,
            decision.PostAecEvidenceFrameCount,
            decision.MeanPostAecResidualRatio,
            decision.MeanAbsoluteRenderCorrelation,
            decision.MeanAbsolutePostAecRenderCorrelation,
            decision.HasDiscontinuity,
            snapshot.AlignmentReady,
            snapshot.EstimatedDelayMs,
            snapshot.DriftPpm,
            snapshot.AlignmentConfidence,
            snapshot.AlignmentGeneration,
            snapshot.RenderStatus,
            snapshot.StreamEpoch,
            snapshot.LastStartSample,
            snapshot.LastEndSampleExclusive,
            batch.StateReset,
            nearEndDiagnostic.CandidateActive,
            nearEndDiagnostic.CandidateAuthorized,
            nearEndDiagnostic.LastCandidateOutcome,
            nearEndDiagnostic.CandidateDecisionEndSample,
            nearEndDiagnostic.MainAudioEndSample,
            nearEndDiagnostic.ActiveTurnId,
            processingState.HasActiveCapture,
            processingState.HasPendingCandidate,
            snapshot.AcousticPathChangeGuardActive,
            snapshot.AcousticPathChangeGuardRemainingMs,
            snapshot.AcousticPathChangeCount,
            snapshot.LastObservationAcousticPathChange,
            snapshot.LastObservationPathChangeGuarded,
            snapshot.LastAcousticPathChangeReason);
    }
}
