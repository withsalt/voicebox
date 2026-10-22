using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Adapters;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;
using VoiceBox.Assistant.Diagnostics.Benchmarking.Models;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Diagnostics.Metrics.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class BenchmarkHooksTests
{
    [Fact]
    public void PlaybackStartupGuard_UsesInjectedTimeProvider()
    {
        var timeProvider = new ManualTimeProvider();
        var guard = new PlaybackStartupGuard(timeProvider);

        guard.Start(500);

        Assert.True(guard.IsActive);
        Assert.Equal(500, guard.GetRemainingMilliseconds());

        timeProvider.Advance(TimeSpan.FromMilliseconds(499));

        Assert.True(guard.IsActive);
        Assert.Equal(1, guard.GetRemainingMilliseconds());

        timeProvider.Advance(TimeSpan.FromMilliseconds(1));

        Assert.False(guard.IsActive);
        Assert.Equal(0, guard.GetRemainingMilliseconds());
    }

    [Fact]
    public void PlaybackStartupGuard_MissingObservationTimestampUsesCurrentClock()
    {
        var timeProvider = new ManualTimeProvider();
        var guard = new PlaybackStartupGuard(timeProvider);
        guard.Start(100);

        timeProvider.Advance(TimeSpan.FromMilliseconds(100));

        Assert.False(guard.IsActiveAt(0));
    }

    [Fact]
    public void AecMetricsSampler_UsesInjectedClockForSamplingCadenceAndMissingTimestamp()
    {
        var provider = new MetricsProvider();
        var timeProvider = new ManualTimeProvider();
        var sampler = new AecMetricsSampler(
            new AcousticProcessingDiagnosticsAdapter(provider),
            NullLogger.Instance,
            timeProvider);
        timeProvider.Advance(TimeSpan.FromMilliseconds(10));

        sampler.SampleIfDue();
        Assert.Equal(1, provider.Calls);
        Assert.True(sampler.TryGetLatest(out AcousticProcessingMetricsSnapshot first));
        Assert.Equal(timeProvider.GetTimestamp(), first.TimestampTicks);

        timeProvider.Advance(TimeSpan.FromMilliseconds(99));
        sampler.SampleIfDue();
        Assert.Equal(1, provider.Calls);

        timeProvider.Advance(TimeSpan.FromMilliseconds(1));
        sampler.SampleIfDue();
        Assert.Equal(2, provider.Calls);
    }

    [Fact]
    public void AecMetricsSampler_ResetForcesFreshSampleAndClearsCachedMetrics()
    {
        var provider = new MetricsProvider();
        var timeProvider = new ManualTimeProvider();
        var sampler = new AecMetricsSampler(new AcousticProcessingDiagnosticsAdapter(provider), NullLogger.Instance, timeProvider);
        timeProvider.Advance(TimeSpan.FromMilliseconds(10));
        sampler.SampleIfDue();
        Assert.True(sampler.TryGetLatest(out _));

        sampler.Reset();

        Assert.False(sampler.TryGetLatest(out _));
        sampler.SampleIfDue();
        Assert.Equal(2, provider.Calls);
        Assert.True(sampler.TryGetLatest(out _));
    }

    [Fact]
    public async Task BenchmarkIngress_RunsCaptureProcessorAndPublishesProductionFrameContracts()
    {
        var processor = new SubtractRenderCaptureProcessor();
        var timeProvider = new ManualTimeProvider();
        using var ingress = new BenchmarkDuplexAudioIngress(
            processor,
            timeProvider,
            sampleRate: 16000,
            channels: 1);
        var eventOrder = new List<string>();
        DuplexAudioFrame duplex = default;
        ProcessedCaptureFrame processed = default;
        float[]? available = null;
        ingress.OnDuplexAudioFrame += OnDuplex;
        ingress.OnProcessedCaptureFrame += OnProcessed;
        ingress.OnDataAvailable += OnAvailable;

        await ingress.StartAsync();
        ingress.SetPlaybackGain(0.5f);
        timeProvider.Advance(TimeSpan.FromMilliseconds(10));
        CaptureRange range = ingress.PushFrame(
            [0.8f, 0.6f, -0.2f],
            [0.3f, 0.1f, 0.2f],
            hasDiscontinuity: true);

        Assert.Equal(2, processor.ResetCalls);
        Assert.Equal(1, processor.ProcessCalls);
        Assert.Equal((uint)1, processor.LastChannels);
        Assert.Equal((uint)16000, processor.LastSampleRate);
        Assert.Equal(["duplex", "processed", "available"], eventOrder);

        Assert.Equal(new float[] { 0.8f, 0.6f, -0.2f }, duplex.CaptureRaw.ToArray());
        Assert.Equal(new float[] { 0.8f, 0.6f, -0.2f }, duplex.CaptureBeforeAec.ToArray());
        Assert.Equal(new float[] { 0.5f, 0.5f, -0.4f }, duplex.CaptureAfterAec.ToArray());
        Assert.Equal(new float[] { 0.3f, 0.1f, 0.2f }, duplex.RenderReference.ToArray());
        Assert.True(duplex.CaptureProcessed);
        Assert.Equal(0.5f, duplex.PlaybackGain);
        Assert.True(duplex.HasDiscontinuity);
        Assert.Equal((uint)3, duplex.FrameCount);
        Assert.Equal(range, duplex.CaptureRange);
        Assert.Equal(range, processed.SourceRange);
        Assert.Equal(duplex.CaptureAfterAec.ToArray(), processed.Audio.ToArray());
        Assert.Equal(duplex.CaptureAfterAec.ToArray(), available);
        Assert.Equal(0, range.StartSample);
        Assert.Equal(3, range.EndSampleExclusive);
        Assert.Equal(
            new RenderSignalTimeline(1, 0, 3, RenderTimelineStatus.Valid),
            duplex.RenderTimeline);

        void OnDuplex(in DuplexAudioFrame frame)
        {
            eventOrder.Add("duplex");
            duplex = frame;
        }

        void OnProcessed(in ProcessedCaptureFrame frame)
        {
            eventOrder.Add("processed");
            processed = frame;
        }

        void OnAvailable(ReadOnlySpan<float> audio)
        {
            eventOrder.Add("available");
            available = audio.ToArray();
        }
    }

    [Fact]
    public async Task BenchmarkIngress_TracksRenderTimelineIndependentlyFromCapture()
    {
        using var ingress = new BenchmarkDuplexAudioIngress(
            new SubtractRenderCaptureProcessor(),
            new ManualTimeProvider(),
            sampleRate: 16000,
            channels: 1);
        var frames = new List<DuplexAudioFrame>();
        ingress.OnDuplexAudioFrame += OnDuplex;

        await ingress.StartAsync();
        ingress.PushFrame([1f, 2f], [0f, 0f]);
        ingress.PushFrame(
            [3f, 4f],
            [0f, 0f],
            renderStatus: RenderTimelineStatus.Gap);
        ingress.PushFrame(
            [5f, 6f],
            [0f, 0f],
            renderStatus: RenderTimelineStatus.Missing,
            hasRenderDiscontinuity: true);

        Assert.Equal(3, frames.Count);
        Assert.Equal(
            new RenderSignalTimeline(1, 0, 2, RenderTimelineStatus.Valid),
            frames[0].RenderTimeline);
        Assert.Equal(
            new RenderSignalTimeline(1, 2, 4, RenderTimelineStatus.Gap),
            frames[1].RenderTimeline);
        RenderSignalTimeline firstRender = frames[0].RenderTimeline;
        RenderSignalTimeline gapRender = frames[1].RenderTimeline;
        Assert.True(firstRender.IsContiguousWith(in gapRender));
        Assert.Equal(
            new RenderSignalTimeline(2, 0, 2, RenderTimelineStatus.Missing, true),
            frames[2].RenderTimeline);
        Assert.All(frames, static frame => Assert.False(frame.HasDiscontinuity));
        Assert.All(frames, static frame => Assert.Equal(1, frame.StreamEpoch));
        Assert.Equal([0L, 2L, 4L], frames.Select(static frame => frame.StartSample));

        void OnDuplex(in DuplexAudioFrame frame) => frames.Add(frame);
    }

    [Fact]
    public async Task BenchmarkIngress_RejectsDiscontinuousRenderGap()
    {
        using var ingress = new BenchmarkDuplexAudioIngress(
            new SubtractRenderCaptureProcessor(),
            new ManualTimeProvider());
        await ingress.StartAsync();

        Assert.Throws<ArgumentException>(() => ingress.PushFrame(
            [0f, 0f],
            [0f, 0f],
            renderStatus: RenderTimelineStatus.Gap,
            hasRenderDiscontinuity: true));
    }

    [Fact]
    public async Task BenchmarkIngress_MapsBufferedProcessorOutputToOriginalCaptureTimeline()
    {
        var processor = new TwoSampleDelayCaptureProcessor();
        using var ingress = new BenchmarkDuplexAudioIngress(
            processor,
            new ManualTimeProvider(),
            sampleRate: 16000,
            channels: 1);
        var frames = new List<(CaptureRange Range, float[] Audio, int Delay)>();
        ingress.OnProcessedCaptureFrame += OnProcessed;

        await ingress.StartAsync();
        ingress.PushFrame([1f, 2f, 3f], [0f, 0f, 0f]);
        ingress.PushFrame([4f, 5f, 6f], [0f, 0f, 0f]);
        ingress.FlushCaptureProcessorDelay();

        Assert.Equal(2, ingress.CaptureProcessorDelaySamples);
        Assert.Equal(3, frames.Count);
        Assert.Equal(new CaptureRange(1, 1, 1, 0, 1, 0), frames[0].Range);
        Assert.Equal(new CaptureRange(1, 1, 2, 1, 4, 0), frames[1].Range);
        Assert.Equal(new CaptureRange(1, 2, 2, 4, 6, 0), frames[2].Range);
        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], frames.SelectMany(static frame => frame.Audio));
        Assert.All(frames, frame => Assert.Equal(2, frame.Delay));

        void OnProcessed(in ProcessedCaptureFrame frame)
            => frames.Add((frame.SourceRange, frame.Audio.ToArray(), frame.DelaySamples));
    }

    [Fact]
    public void BenchmarkProbe_ObservesWithoutManufacturingGuardedDecisions()
    {
        var probe = new VoiceAssistantBenchmarkProbe();
        Assert.True(probe.InterruptionAdmissionsSettled);
        probe.AttachInterruptionAdmissionView(new FixedAdmissionProgress(false));
        Assert.False(probe.InterruptionAdmissionsSettled);
        probe.AttachInterruptionAdmissionView(new FixedAdmissionProgress(true));
        Assert.True(probe.InterruptionAdmissionsSettled);
        RenderAwareBargeInSnapshot probability = Detector(0.45f, 0.70f);
        BargeInDecision guardedDecision = Decision(frameCount: 4, peakScoreDurationMs: 80);

        probe.Observe(
            guardedDecision,
            probability,
            startupGuardActive: true,
            playbackActive: true,
            coordinator: default,
            assistantState: AssistantState.Speaking,
            isSpeakingWithoutPendingCandidate: true);
        VoiceAssistantBenchmarkSnapshot guarded = probe.GetSnapshot();

        Assert.Equal(default, guarded.Decision);
        Assert.Equal(0.45f, guarded.PeakNearEndProbability);
        Assert.Equal(0.70f, guarded.PeakSmoothedNearEndProbability);

        BargeInDecision productionDecision = Decision(frameCount: 8, peakScoreDurationMs: 160);
        probe.Observe(
            productionDecision,
            Detector(0.35f, 0.55f),
            startupGuardActive: false,
            playbackActive: true,
            coordinator: default,
            assistantState: AssistantState.Speaking,
            isSpeakingWithoutPendingCandidate: true);
        probe.Observe(
            Decision(frameCount: 9, peakScoreDurationMs: 120),
            Detector(0.20f, 0.30f, RenderTimelineStatus.Unavailable),
            startupGuardActive: false,
            playbackActive: false,
            coordinator: default,
            assistantState: AssistantState.Idle,
            isSpeakingWithoutPendingCandidate: false);
        BargeInDecision equalPeakLatest = Decision(frameCount: 10, peakScoreDurationMs: 160);
        probe.Observe(
            equalPeakLatest,
            Detector(0.30f, 0.40f),
            startupGuardActive: false,
            playbackActive: true,
            coordinator: default,
            assistantState: AssistantState.Speaking,
            isSpeakingWithoutPendingCandidate: true);

        VoiceAssistantBenchmarkSnapshot snapshot = probe.GetSnapshot();
        Assert.Equal(equalPeakLatest, snapshot.Decision);
        Assert.Equal(0.45f, snapshot.PeakNearEndProbability);
        Assert.Equal(0.70f, snapshot.PeakSmoothedNearEndProbability);
        Assert.Equal(AssistantState.Speaking, snapshot.AssistantState);
        Assert.True(snapshot.IsSpeakingWithoutPendingCandidate);
        Assert.True(snapshot.PlaybackActive);
        Assert.True(snapshot.RenderActive);
    }

    [Fact]
    public void BenchmarkProbe_ReportsPendingConfirmationWithoutParticipatingInDecisionMaking()
    {
        var probe = new VoiceAssistantBenchmarkProbe();
        var pending = new ActiveNearEndDiagnosticSnapshot(
            CandidateActive: false,
            CandidateAuthorized: false,
            CandidateDecisionEndSample: 4800,
            MainAudioEndSample: 4800,
            ActiveTurnId: 0,
            LastCandidateOutcome: "queued:confirmation",
            ConfirmationPending: true);

        probe.Observe(
            Decision(frameCount: 8, peakScoreDurationMs: 160),
            Detector(0.35f, 0.55f),
            startupGuardActive: false,
            playbackActive: true,
            coordinator: pending,
            assistantState: AssistantState.Speaking,
            isSpeakingWithoutPendingCandidate: true);

        VoiceAssistantBenchmarkSnapshot snapshot = probe.GetSnapshot();

        Assert.True(snapshot.Coordinator.ConfirmationPending);
        Assert.Equal("queued:confirmation", snapshot.Coordinator.LastCandidateOutcome);
        Assert.Equal(BargeInAction.Continue, snapshot.Decision.Action);

        probe.ObserveControlPlaneProgress(
            pending with { ConfirmationPending = false },
            AssistantState.Listening,
            isSpeakingWithoutPendingCandidate: false);

        snapshot = probe.GetSnapshot();

        Assert.False(snapshot.Coordinator.ConfirmationPending);
        Assert.Equal(BargeInAction.Continue, snapshot.Decision.Action);
    }

    [Fact]
    public void BenchmarkProbe_ControlTrace_IsPassiveAndCaptureTimestamped()
    {
        var probe = new VoiceAssistantBenchmarkProbe();
        probe.AttachCapturePosition(new FixedCapturePosition(1600));

        probe.ObserveControlTrace(
            "effect.clear-playback",
            turnId: 2,
            operationId: 7,
            detail: "diagnostic");

        ControlTraceEntry[] trace = probe.GetControlTrace();
        var entry = Assert.Single(trace);
        Assert.Equal("effect.clear-playback", entry.Stage);
        Assert.Equal(1600, entry.CapturePosition);
        Assert.Equal(2, entry.TurnId);
        Assert.Equal(7, entry.OperationId);
        Assert.Equal("diagnostic", entry.Detail);
    }

    private static RenderAwareBargeInSnapshot Detector(
        float instant,
        float smoothed,
        RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid)
        => new(
            HasObservation: true,
            LatestProbability: instant,
            LatestSmoothedProbability: smoothed,
            LatestNearEndSpeech: smoothed >= 0.5f,
            AlignmentReady: true,
            EstimatedDelayMs: 0,
            DriftPpm: 0,
            AlignmentConfidence: 1,
            AlignmentGeneration: 1,
            StreamEpoch: 1,
            LastStartSample: 0,
            LastEndSampleExclusive: 160,
            RenderStatus: renderStatus,
            IsFaulted: false);

    private static BargeInDecision Decision(int frameCount, double peakScoreDurationMs)
        => new(
            Action: BargeInAction.Continue,
            FrameCount: frameCount,
            AcceptedFrameCount: frameCount,
            RejectedFrameCount: 0,
            EchoLikeFrameCount: 0,
            TotalDurationMs: frameCount * 20,
            AcceptedDurationMs: frameCount * 20,
            RequiredDurationMs: 280,
            ScoreDurationMs: peakScoreDurationMs,
            PeakScoreDurationMs: peakScoreDurationMs,
            LastFrameRms: 0.1,
            LastRequiredRms: 0.05,
            MinRequiredRms: 0.05,
            MaxRequiredRms: 0.05,
            Reason: "test");

    private sealed class ManualTimeProvider : TimeProvider
    {
        private long _timestamp;

        public override long TimestampFrequency => TimeSpan.TicksPerSecond;

        public override long GetTimestamp() => _timestamp;

        public void Advance(TimeSpan duration)
            => _timestamp = checked(_timestamp + duration.Ticks);
    }

    private sealed class SubtractRenderCaptureProcessor : IAcousticEchoCancellationService
    {
        public int ProcessCalls { get; private set; }
        public int ResetCalls { get; private set; }
        public uint LastChannels { get; private set; }
        public uint LastSampleRate { get; private set; }
        public int DelaySamples => 0;

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate)
        {
            ProcessCalls++;
            LastChannels = channels;
            LastSampleRate = sampleRate;
            for (int i = 0; i < capture.Length; i++)
                capture[i] -= render[i];
        }

        public void Reset() => ResetCalls++;

        public void Dispose()
        {
        }
    }

    private sealed class TwoSampleDelayCaptureProcessor : IAcousticEchoCancellationService
    {
        private readonly Queue<float> _delay = new();

        public int DelaySamples => 2;

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate)
        {
            for (int i = 0; i < capture.Length; i++)
            {
                float input = capture[i];
                capture[i] = _delay.Count < DelaySamples
                    ? 0
                    : _delay.Dequeue();
                _delay.Enqueue(input);
            }
        }

        public void Reset() => _delay.Clear();

        public void Dispose()
        {
        }
    }

    private sealed class MetricsProvider : IAcousticProcessingMetricsSource
    {
        public int Calls { get; private set; }

        public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
        {
            Calls++;
            metrics = new AcousticProcessingMetrics(
                HasEchoReturnLoss: false,
                EchoReturnLoss: 0,
                HasEchoReturnLossEnhancement: false,
                EchoReturnLossEnhancement: 0,
                HasDelayMs: false,
                DelayMs: 0,
                HasDelayMedianMs: false,
                DelayMedianMs: 0,
                HasResidualEchoLikelihood: false,
                ResidualEchoLikelihood: 0);
            return true;
        }
    }

    private sealed class FixedAdmissionProgress(bool settled) : IInterruptionAdmissionProgressView
    {
        public bool HasSettledAllInterruptionAdmissions { get; } = settled;
    }

    private sealed class FixedCapturePosition(long position) : ICapturePositionView
    {
        public long CurrentCaptureSamplePosition { get; } = position;
    }
}
