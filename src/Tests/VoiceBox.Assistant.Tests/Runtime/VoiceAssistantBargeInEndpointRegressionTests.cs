using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.State;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData("Match", 13760)]
    [InlineData("Mismatch", 14080)]
    [InlineData("InsufficientEvidence", 14080)]
    [InlineData("Unavailable", 14080)]
    public async Task ActiveNearEndTrigger_RejectionPreservesUtteranceOnset(string firstStatus, int expectedSamples)
    {
        var speaker = new SequencedSpeakerEvidenceProvider(Enum.Parse<SpeakerVerificationStatus>(firstStatus),
            SpeakerVerificationStatus.Match);
        var fixture = new EndpointRegressionFixture(speaker, 13760);
        fixture.Push();
        if (firstStatus != "Match")
        {
            Assert.Equal(1, fixture.Harness.CommandDeniedCount);
            Assert.True(fixture.Harness.TryLocate(fixture.Audio.AsSpan(0, 320), out _, out _, out _));
            await Task.Delay(120); // Exercise the default 100 ms cooldown without advancing PCM.
            fixture.Append();
            fixture.Push();
        }
        Assert.True(fixture.Harness.TryAcquire(1, out var lease));
        Assert.Equal(expectedSamples, lease.InitialAudio.Sum(chunk => chunk.Length));
        Assert.Equal(0, lease.Selection.SelectedStartSample);
        Assert.Equal(fixture.Audio, lease.InitialAudio.SelectMany(chunk => chunk).Take(13760));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActiveNearEndCapture_EndpointBeforeSnapshotDoesNotStrandCapture(bool extraFrame, bool processEndpointBeforeCommit)
    {
        var fixture = new EndpointRegressionFixture(new SequencedSpeakerEvidenceProvider(
            SpeakerVerificationStatus.Pending, SpeakerVerificationStatus.Match), 9600);
        fixture.Push();
        fixture.Harness.ObserveUnownedEndpointSpeechEnded();
        if (extraFrame)
            fixture.Append();
        fixture.Push();
        Assert.True(fixture.Harness.TryAcquire(1, out var lease));
        Assert.False(lease.EndpointVadCandidate);
        Assert.Equal("main-vad-timeline", lease.Selection.Source);
        Assert.Equal(0, lease.Selection.SelectedStartSample);
        Assert.Equal(extraFrame ? 9920 : 9600, lease.Selection.SelectedEndSample);
        if (processEndpointBeforeCommit)
            fixture.Harness.ObserveEndpointSpeechEnded();
        fixture.Harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        if (!processEndpointBeforeCommit)
            fixture.Harness.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint: true);
        if (!extraFrame)
        {
            Assert.Equal(1, fixture.Ended);
            Assert.Equal(0, fixture.Harness.GetDiagnosticSnapshot().ActiveTurnId);
            return;
        }
        fixture.SpeechActive = false;
        for (int frame = 1; frame <= 26; frame++)
        {
            fixture.Append(feed: true);
            fixture.Push();
            if (frame < 26)
                Assert.Equal(0, fixture.Ended);
        }
        Assert.Equal(1, fixture.Ended);
        Assert.Equal(0, fixture.Harness.GetDiagnosticSnapshot().ActiveTurnId);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ActiveNearEndCapture_EndpointAfterSnapshotCannotRestoreStaleVadSpeech(bool processEndpointBeforeCommit)
    {
        var fixture = new EndpointRegressionFixture(new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match), 9600);
        fixture.Push();
        Assert.True(fixture.Harness.TryAcquire(1, out var lease));
        Assert.True(lease.EndpointVadCandidate);
        fixture.Harness.ObserveUnownedEndpointSpeechEnded();
        fixture.Append();
        if (processEndpointBeforeCommit)
            fixture.Harness.ObserveEndpointSpeechEnded();
        fixture.Harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        if (!processEndpointBeforeCommit)
            fixture.Harness.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint: true);
        Assert.Equal(1, fixture.Ended);
        Assert.Equal(0, fixture.Harness.GetDiagnosticSnapshot().ActiveTurnId);
    }

    [Fact]
    public void ActiveNearEndCapture_FreshVadOnsetSupersedesQueuedEndpoint()
    {
        var fixture = new EndpointRegressionFixture(new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match), 9600);
        fixture.Push();
        Assert.True(fixture.Harness.TryAcquire(1, out var lease));
        fixture.Harness.ObserveUnownedEndpointSpeechEnded();
        fixture.Harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        fixture.Harness.ObserveEndpointSpeechStarted();
        fixture.Harness.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint: true);
        Assert.Equal(0, fixture.Ended);
        Assert.Equal(1, fixture.Harness.GetDiagnosticSnapshot().ActiveTurnId);
        fixture.Harness.ObserveEndpointSpeechEnded();
        Assert.Equal(1, fixture.Ended);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ActiveNearEndCapture_VadEndpointDrainsWithoutAdditionalSilence(bool pendingAudio)
    {
        var fixture = new EndpointRegressionFixture(new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match), 9600);
        fixture.Push();
        Assert.True(fixture.Harness.TryAcquire(1, out var lease));
        fixture.Harness.ApplyConfirmedCapture(lease, CreateConfirmedDecision());
        if (pendingAudio)
        {
            fixture.Harness.DataPlane.BeginEndpointObservation();
            MainAudioObservation tail = fixture.Append();
            fixture.Harness.ObserveEndpointSpeechEnded();
            Assert.Equal(0, fixture.Ended);
            fixture.Harness.FeedFromRecognitionAudio(new float[320], tail);
            Assert.Equal(0, fixture.Ended);
            fixture.Harness.DataPlane.CompleteEndpointObservation();
        }
        else
        {
            fixture.Harness.ObserveEndpointSpeechEnded();
        }
        Assert.Equal(1, fixture.Ended);
        Assert.Equal(0, fixture.Harness.GetDiagnosticSnapshot().ActiveTurnId);
    }

    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public async Task VadSpeakerGate_EachWindowGetsItsOwnInferenceBudget(int requiredMatches, bool pollBetweenWindows)
    {
        var clock = new ManualTimerTimeProvider();
        var provider = new ScriptedSpeakerService(windowSamples: 8000, hopSamples: 3200);
        using var verifier = CreateDeadlineVerifier(provider, requiredMatches);
        using var speaker = new DeadlineSpeakerModule(verifier);
        var timeline = new GateTimelineLocator();
        var gate = new VadSpeakerGateCoordinator(speaker, new TestGateHost(), timeline, NullLogger.Instance, clock);
        gate.BeginCandidate(SpeakerVerificationPurpose.BargeInCommand);
        try
        {
            long start = 0;
            for (int window = 0; window < requiredMatches; window++)
            {
                if (window > 0)
                    clock.Advance(TimeSpan.FromMilliseconds(51));
                provider.ExtractEntered = new(TaskCreationOptions.RunContinuationsAsynchronously);
                provider.ExtractBlock = new(TaskCreationOptions.RunContinuationsAsynchronously);
                float[] audio = ScriptedSpeakerService.OwnerScoreAudio(0.6f, window == 0 ? 8000 : 3200);
                verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, start, audio));
                timeline.Start = start;
                Assert.Equal(BargeInSpeakerGateStatus.Pending, gate.Evaluate(audio));
                await provider.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
                Assert.Equal(SpeakerVerificationReason.WaitingForInference, speaker.Last.Reason);
                Assert.Equal(window * 3200, speaker.Last.PendingWindowStartSample);
                Assert.Equal(window, speaker.Last.MatchedWindows);

                clock.Advance(TimeSpan.FromMilliseconds(149));
                Assert.Equal(BargeInSpeakerGateStatus.Pending, gate.Evaluate(audio));
                provider.ExtractBlock.TrySetResult();
                int expectedWindows = window + 1;
                await WaitUntilAsync(() => verifier.Cache.Count >= expectedWindows);
                if (pollBetweenWindows || expectedWindows == requiredMatches)
                {
                    Assert.Equal(expectedWindows == requiredMatches ? BargeInSpeakerGateStatus.Match : BargeInSpeakerGateStatus.Pending,
                        gate.Evaluate(audio));
                }
                start += audio.Length;
            }
            Assert.Equal(requiredMatches, speaker.Last.MatchedWindows);
        }
        finally
        {
            provider.ExtractBlock?.TrySetResult();
        }
    }

    [Fact]
    public async Task VadSpeakerGate_SlidingWindowsDoNotRenewABlockedInferenceDeadline()
    {
        var clock = new ManualTimerTimeProvider();
        var provider = new ScriptedSpeakerService(windowSamples: 8000, hopSamples: 3200)
        {
            ExtractEntered = new(TaskCreationOptions.RunContinuationsAsynchronously),
            ExtractBlock = new(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        using var verifier = CreateDeadlineVerifier(provider, requiredMatches: 2);
        using var speaker = new DeadlineSpeakerModule(verifier);
        var timeline = new GateTimelineLocator();
        var gate = new VadSpeakerGateCoordinator(speaker, new TestGateHost(), timeline, NullLogger.Instance, clock);
        gate.BeginCandidate(SpeakerVerificationPurpose.BargeInCommand);
        try
        {
            float[] audio = ScriptedSpeakerService.OwnerScoreAudio(0.6f, 8000);
            verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, 0, audio));
            Assert.Equal(BargeInSpeakerGateStatus.Pending, gate.Evaluate(audio));
            await provider.ExtractEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            for (int hop = 0; hop < 4; hop++)
            {
                timeline.Start = 8000 + hop * 3200;
                audio = ScriptedSpeakerService.OwnerScoreAudio(0.6f, 3200);
                verifier.OnSpeakerCapture(SpeakerTestProfiles.Frame(1, timeline.Start, audio));
                clock.Advance(TimeSpan.FromMilliseconds(40));
                Assert.Equal(hop == 3 ? BargeInSpeakerGateStatus.Rejected : BargeInSpeakerGateStatus.Pending,
                    gate.Evaluate(audio));
                Assert.Equal(0, speaker.Last.PendingWindowStartSample);
            }
            Assert.Equal(1, provider.ExtractCount);
            Assert.Equal(SpeakerVerificationStatus.Unavailable, gate.LastDeniedStatus);
        }
        finally
        {
            provider.ExtractBlock.TrySetResult();
        }
    }

    private sealed class EndpointRegressionFixture
    {
        public NearEndTestHarness Harness { get; }
        public float[] Audio { get; }
        public long End;
        public bool SpeechActive = true;
        public int Ended;
        private long _sequence = 1;

        public EndpointRegressionFixture(ISpeakerEvidenceProvider speaker, int samples)
        {
            var services = TestServices.Create();
            Audio = Enumerable.Range(0, samples).Select(i => 0.03f + i * 0.00001f).ToArray();
            End = samples;
            Harness = CreateNearEndTestHarness(new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
            }, new TestRecognitionAudioPort(services.Asr), () => true,
                (_, _, _) => new BargeInDecision(BargeInAction.Continue,
                    1, SpeechActive ? 1 : 0, 0, 0, 20, SpeechActive ? 20 : 0, 180, 20, 20,
                    0.1, 0, 0, 0, "endpoint-regression", StreamEpoch: 1,
                    StartSample: 0, EndSampleExclusive: End,
                    LatestFrameAccepted: SpeechActive, FirstAcceptedStartSample: 0),
                new AssistantAudioFormat(new DuplexAssistantAudioFrontend(services.Audio).Input,
                    new SpeechSynthesisStreamAdapter(services.Tts, null, null)), NullLogger.Instance,
                () => false, () => true, _ => true, (_, _) => Ended++, speaker);
            ObserveRecognitionAudio(Harness, 0, Audio);
            Harness.ObserveEndpointSpeechStarted();
            Harness.ObserveEndpointSpeechAudio(Audio);
        }

        public MainAudioObservation Append(bool feed = false)
        {
            float[] audio = new float[320];
            var frame = new ProcessedCaptureFrame(new CaptureRange(1, ++_sequence, _sequence,
                End, End + 320, _sequence), 16000, 1, audio, 0, false);
            MainAudioObservation observed = Harness.ObserveRecognitionAudio(frame);
            End += 320;
            if (feed)
                Harness.FeedFromRecognitionAudio(audio, observed);
            return observed;
        }

        public void Push() => Harness.ObserveDuplexAudio(CreateDuplexFrame(_sequence, End - 320, 320), false);
    }

    private sealed class DeadlineSpeakerModule : ISpeakerModule
    {
        private readonly SpeechGatedSpeakerVerifier _verifier;
        private readonly SpeakerVerificationState _state;
        public DeadlineSpeakerModule(SpeechGatedSpeakerVerifier verifier)
        {
            _verifier = verifier;
            _state = new SpeakerVerificationState(this);
        }
        public SpeakerVerificationResult Last { get; private set; }
        public SpeakerVerificationPolicy Policy => _verifier.Policy;
        public SpeakerCaptureDescriptor Descriptor => SpeakerTestProfiles.Descriptor();
        public AssistantTimelineRequirement TimelineRequirement => AssistantTimelineRequirement.SpeakerCaptureTimeline;
        public bool IsCandidateActive => _state.IsCandidateActive;
        public long CurrentVersion => _state.CurrentVersion;
        public void BeginCandidate(SpeakerVerificationPurpose purpose) => _state.BeginCandidate(purpose);
        public void Reset() => _state.Reset();
        public bool TryBuildQuery(long version, long epoch, long start, long end, long ticks, out SpeakerEvidenceQuery query)
            => _state.TryBuildQuery(version, epoch, start, end, ticks, out query);
        public bool TryCompleteQuery(long version, long ticks, out SpeakerEvidenceQuery query)
            => _state.TryCompleteQuery(version, ticks, out query);
        public bool IsCurrent(long version) => _state.IsCurrent(version);
        public bool TryResolve(long version) => _state.TryResolve(version);
        public SpeakerCandidateId NewCandidateId() => _verifier.NewCandidateId();
        public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query) => Last = _verifier.Evaluate(in query);
        public Task<SpeakerVerificationResult> WaitForDecisionAsync(SpeakerEvidenceQuery query, TimeSpan timeout, CancellationToken cancellationToken = default)
            => _verifier.WaitForDecisionAsync(query, timeout, cancellationToken);
        public void Cancel(SpeakerCandidateId id) => _verifier.Cancel(id);
        public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => _verifier.GetInferenceTimeout(purpose);
        public void OnSpeakerCapture(in SpeakerCaptureFrame frame) => _verifier.OnSpeakerCapture(in frame);
        public void SetPlaybackActive(bool active) { }
        public void Start() { }
        public Task StopAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public void Dispose() { }
    }

    private sealed class GateTimelineLocator : ISpeechTimelineLocator
    {
        public long Start;
        public bool TryLocate(ReadOnlySpan<float> audio, out long epoch, out long start, out long end)
        {
            epoch = 1;
            start = Start;
            end = Start + audio.Length;
            return true;
        }
    }

    private sealed class TestGateHost : IVadSpeakerGateHost
    {
        public CancellationToken RunToken => CancellationToken.None;
        public bool IsRunCancelled => false;
        public long NextOperationId() => 1;
        public void SubmitGateResolved(long id, SpeakerVerificationPurpose purpose, BargeInSpeakerGateStatus status, TimeSpan? pendingSpeechEnd) { }
        public SpeechBoundaryObservation SpeechEndedWithoutSpeakerGate(TimeSpan duration) => default;
        public void Track(string name, Task task) { }
    }
}
