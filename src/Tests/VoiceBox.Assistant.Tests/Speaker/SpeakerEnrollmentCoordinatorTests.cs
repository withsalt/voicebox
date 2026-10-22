using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.Assistant.Modules.Speaker.Exceptions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;
using Xunit;

#pragma warning disable CS0067

namespace VoiceBox.Assistant.Tests;

public sealed class SpeakerEnrollmentCoordinatorTests : IDisposable
{
    private const int FrameSamples = 1600;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "voicebox-enroll", Guid.NewGuid().ToString("N"));
    private readonly ScriptedSpeakerService _speaker = new();
    private readonly EnrollmentDuplexService _audio = new();
    private readonly EnergyVadService _vad = new();
    private readonly FakeStimulusPlayer _stimulus = new();

    public SpeakerEnrollmentCoordinatorTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public async Task CleanOnlyEnrollment_WritesV2VariantWithCleanBankAndIndependentConfirmation()
    {
        VoiceAssistantSpeakerOptions options = Options();
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio,
            _vad,
            _speaker,
            options,
            new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 },
            logger: NullLogger<SpeakerEnrollmentCoordinator>.Instance);
        var progress = new List<SpeakerEnrollmentProgressEventArgs>();
        coordinator.ProgressChanged += (_, e) => { lock (progress) progress.Add(e); };
        DriveSegments(coordinator, _ => 0.9f);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.True(File.Exists(result.ProfilePath));
        Assert.Equal(coordinator.FrontendFingerprint, result.FrontendFingerprint);
        Assert.True(result.CleanPrototypeCount >= 2);
        Assert.Equal(0, result.DoubleTalkPrototypeCount);
        Assert.Equal(5, result.CleanSegmentsRetained);
        Assert.Equal(0, result.RetriedSegments);
        Assert.True(result.ConfirmationMinScore >= 0.5);

        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, result.ProfilePath);
        OwnerSpeakerProfileVariant variant = Assert.Single(profile.Variants);
        Assert.Equal(SpeakerCaptureDescriptor.PostCaptureProcessorTapId, variant.TapId);
        Assert.True(variant.Banks.ContainsKey(SpeakerPurposeBank.CleanNearEnd));
        Assert.False(variant.Banks.ContainsKey(SpeakerPurposeBank.DoubleTalkNearEnd));
        Assert.True(coordinator.HasUsableProfile());
        Assert.False(coordinator.HasUsableProfile(requireDoubleTalkBank: true));
        Assert.Equal(0, _audio.StartCount - _audio.StopCount);
        Assert.Contains(progress, e => e.Stage == SpeakerEnrollmentStage.CleanConfirmation);
        Assert.Contains(progress, e => e.Stage == SpeakerEnrollmentStage.Completed);
        Assert.DoesNotContain(progress, e => e.Stage == SpeakerEnrollmentStage.DoubleTalkEnrollment);
        // Collecting notifications carry the segment they belong to, so a UI can show "segment 3/5".
        Assert.Contains(progress, e => e.Kind == SpeakerEnrollmentProgressKind.Collecting && e.SegmentIndex == 4 && e.SegmentCount == 5 && e.Attempt == 1);
        Assert.Equal(5, progress.Count(e => e.Kind == SpeakerEnrollmentProgressKind.SegmentOutcome && e.ReasonCode == SpeakerEnrollmentReasonCode.SegmentAccepted && e.Stage == SpeakerEnrollmentStage.CleanEnrollment));
        // Prototype embeddings come from enrollment segments only, never from the confirmation.
        Assert.Equal(5, _audio.SegmentsRecorded - 1);
    }

    [Fact]
    public async Task LowEnergySegment_IsRetriedAloneWithTypedReason()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        var reasons = new List<SpeakerEnrollmentReasonCode>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.ReasonCode != SpeakerEnrollmentReasonCode.None)
                lock (reasons) reasons.Add(e.ReasonCode);
        };
        // Second segment first attempt is far too quiet, then normal.
        DriveSegments(coordinator, announce => announce.SegmentIndex == 1 && announce.Attempt == 1 ? 0.002f : 0.9f);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(1, result.RetriedSegments);
        Assert.Equal(5, result.CleanSegmentsRetained);
        Assert.Contains(SpeakerEnrollmentReasonCode.LowEnergy, reasons);
        Assert.Equal(1, reasons.Count(r => r != SpeakerEnrollmentReasonCode.SegmentAccepted));
    }

    [Fact]
    public async Task TimelineGap_RetriesOnlyTheCurrentSegment()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        var reasons = new List<SpeakerEnrollmentReasonCode>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.ReasonCode != SpeakerEnrollmentReasonCode.None)
                lock (reasons) reasons.Add(e.ReasonCode);
        };
        DriveSegments(
            coordinator,
            _ => 0.9f,
            gapOn: announce => announce.SegmentIndex == 2 && announce.Attempt == 1);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(1, result.RetriedSegments);
        Assert.Contains(SpeakerEnrollmentReasonCode.TimelineGap, reasons);
    }

    [Fact]
    public async Task OutlierSegment_IsDiscardedAndDoesNotEnterPrototypes()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0, CleanSegmentCount = 6, MinCleanSegmentsRetained = 4 });
        // Segment 3 is another voice (owner score 0.05, still loud enough to pass the energy gate).
        DriveSegments(coordinator, announce => announce.Stage == SpeakerEnrollmentStage.CleanEnrollment && announce.SegmentIndex == 3 ? 0.05f : 0.9f);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(5, result.CleanSegmentsRetained);
        Assert.Equal(1, result.DiscardedSegments);
        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, result.ProfilePath);
        SpeakerPrototypeBank bank = profile.Variants[0].Banks[SpeakerPurposeBank.CleanNearEnd];
        foreach (SpeakerPrototype prototype in bank.Prototypes)
            Assert.True(_speaker.CompareEmbeddings(prototype.Embedding, ScriptedSpeakerService.OwnerAxis()) > 0.8f);
    }

    [Fact]
    public async Task NotEnoughRetainedSegments_FailsWithoutWritingProfile()
    {
        VoiceAssistantSpeakerOptions options = Options();
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, options, new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0, CleanSegmentCount = 5, MinCleanSegmentsRetained = 5 });
        DriveSegments(coordinator, announce => announce.SegmentIndex == 4 ? 0.05f : 0.9f);

        SpeakerEnrollmentException error = await Assert.ThrowsAsync<SpeakerEnrollmentException>(
            () => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(SpeakerEnrollmentReasonCode.NotEnoughSegments, error.Reason);
        Assert.False(File.Exists(options.ResolveProfilePath()));
        Assert.Equal(0, _audio.StartCount - _audio.StopCount);
    }

    [Fact]
    public async Task ConfirmationMismatch_DoesNotWriteProfile()
    {
        VoiceAssistantSpeakerOptions options = Options();
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, options, new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        DriveSegments(coordinator, announce => announce.Stage == SpeakerEnrollmentStage.CleanConfirmation ? 0.05f : 0.9f);

        SpeakerEnrollmentException error = await Assert.ThrowsAsync<SpeakerEnrollmentException>(
            () => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(SpeakerEnrollmentReasonCode.ConfirmationFailed, error.Reason);
        Assert.False(File.Exists(options.ResolveProfilePath()));
    }

    [Fact]
    public async Task DoubleTalkEnrollment_RequiresStimulusPlayer()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(_audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions());
        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() => coordinator.EnrollAsync());
        Assert.Contains("ISpeakerEnrollmentStimulusPlayer", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DoubleTalkEnrollment_PlaysStimulusRequiresRenderAndWritesBothBanks()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio,
            _vad,
            _speaker,
            Options(),
            new SpeakerEnrollmentOptions { CleanSegmentCount = 4, MinCleanSegmentsRetained = 4, DoubleTalkSegmentCount = 3, MinDoubleTalkSegmentsRetained = 2 },
            _stimulus);
        var reasons = new List<SpeakerEnrollmentReasonCode>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.ReasonCode != SpeakerEnrollmentReasonCode.None)
                lock (reasons) reasons.Add(e.ReasonCode);
        };
        // The first double-talk attempt has no render (stimulus silent), so it must be retried.
        _stimulus.SilentOnFirstStart = true;
        DriveSegments(coordinator, _ => 0.85f);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(15));

        Assert.True(result.DoubleTalkPrototypeCount >= 2);
        Assert.Equal(3, result.DoubleTalkSegmentsRetained);
        Assert.Contains(SpeakerEnrollmentReasonCode.NoRenderDuringDoubleTalk, reasons);
        Assert.True(_stimulus.StartCount >= 4);
        Assert.Equal(_stimulus.StartCount, _stimulus.StopCount);
        Assert.True(coordinator.HasUsableProfile(requireDoubleTalkBank: true));
    }

    [Fact]
    public async Task Enrollment_OverwritesExistingProfileFile()
    {
        VoiceAssistantSpeakerOptions options = Options();
        SpeakerCaptureDescriptor other = SpeakerTestProfiles.Descriptor(aec: true);
        await SpeakerProfileStore.SaveAsync(
            _speaker,
            options.ResolveProfilePath(),
            SpeakerTestProfiles.Profile(_speaker, other, includeDoubleTalk: false));

        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, options, new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        DriveSegments(coordinator, _ => 0.9f);
        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, result.ProfilePath);
        OwnerSpeakerProfileVariant variant = Assert.Single(profile.Variants);
        Assert.Equal(coordinator.FrontendFingerprint, variant.FrontendFingerprint);
        Assert.Null(profile.FindVariant(other.FrontendFingerprint));
    }

    [Fact]
    public async Task Enrollment_ReplacesRetiredFormat5Profile()
    {
        VoiceAssistantSpeakerOptions options = Options();
        await File.WriteAllTextAsync(
            options.ResolveProfilePath(),
            """
            {"formatVersion":5,"speakerId":"owner","serviceType":"legacy","sampleRate":16000,"embeddingSize":8,"streamingWindowSamples":8000,"streamingHopSamples":1600,"audioTap":"processed-capture-v1","embedding":"AAAA"}
            """);

        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, options, new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        DriveSegments(coordinator, _ => 0.9f);

        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(_speaker, result.ProfilePath);
        Assert.Equal(SpeakerCaptureDescriptor.PostCaptureProcessorTapId, Assert.Single(profile.Variants).TapId);
        string json = await File.ReadAllTextAsync(result.ProfilePath);
        Assert.Contains("owner-speaker-profile-v2", json, StringComparison.Ordinal);
        Assert.DoesNotContain("processed-capture-v1", json, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TooManyRetries_OnLaterCleanSegment_ResumeKeepsEarlierSegments()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio,
            _vad,
            _speaker,
            Options(),
            new SpeakerEnrollmentOptions { CleanSegmentCount = 3, MinCleanSegmentsRetained = 3, DoubleTalkSegmentCount = 0, MaxAttemptsPerSegment = 2 });
        int pass = 0;
        var started = new List<SpeakerEnrollmentProgressEventArgs>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.Kind == SpeakerEnrollmentProgressKind.SegmentStarted)
                lock (started) started.Add(e);
        };
        DriveSegments(coordinator, e => pass == 0 && e.Stage == SpeakerEnrollmentStage.CleanEnrollment && e.SegmentIndex == 2 ? 0.002f : 0.9f);

        SpeakerEnrollmentException error = await Assert.ThrowsAsync<SpeakerEnrollmentException>(
            () => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(SpeakerEnrollmentReasonCode.TooManyRetries, error.Reason);
        Assert.Equal(SpeakerEnrollmentStage.CleanEnrollment, error.Stage);
        Assert.Equal(2, error.SegmentIndex);
        Assert.True(coordinator.PendingResume.HasValue);
        SpeakerEnrollmentResumeInfo resume = coordinator.PendingResume.Value;
        Assert.Equal(SpeakerEnrollmentStage.CleanEnrollment, resume.Stage);
        Assert.Equal(2, resume.SegmentIndex);
        Assert.Equal(2, resume.AcceptedCleanSegments);

        pass = 1;
        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(3, result.CleanSegmentsRetained);
        Assert.Null(coordinator.PendingResume);
        lock (started)
        {
            Assert.Equal(1, started.Count(e => e.Stage == SpeakerEnrollmentStage.CleanEnrollment && e.SegmentIndex == 0));
            Assert.Equal(1, started.Count(e => e.Stage == SpeakerEnrollmentStage.CleanEnrollment && e.SegmentIndex == 1));
            Assert.Contains(started, e => e.Stage == SpeakerEnrollmentStage.CleanEnrollment && e.SegmentIndex == 2);
        }
    }

    [Fact]
    public async Task ConfirmationFailed_ResumeRetriesConfirmationOnly()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        int pass = 0;
        var started = new List<SpeakerEnrollmentProgressEventArgs>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.Kind == SpeakerEnrollmentProgressKind.SegmentStarted)
                lock (started) started.Add(e);
        };
        DriveSegments(coordinator, e => pass == 0 && e.Stage == SpeakerEnrollmentStage.CleanConfirmation ? 0.05f : 0.9f);

        SpeakerEnrollmentException error = await Assert.ThrowsAsync<SpeakerEnrollmentException>(
            () => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Equal(SpeakerEnrollmentReasonCode.ConfirmationFailed, error.Reason);
        Assert.True(coordinator.PendingResume.HasValue);
        SpeakerEnrollmentResumeInfo resume = coordinator.PendingResume.Value;
        Assert.Equal(SpeakerEnrollmentStage.CleanConfirmation, resume.Stage);
        Assert.Equal(5, resume.AcceptedCleanSegments);

        pass = 1;
        started.Clear();
        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));

        Assert.True(result.ConfirmationMinScore >= 0.5);
        lock (started)
        {
            Assert.DoesNotContain(started, e => e.Stage == SpeakerEnrollmentStage.CleanEnrollment);
            Assert.Contains(started, e => e.Stage == SpeakerEnrollmentStage.CleanConfirmation);
        }
        Assert.Null(coordinator.PendingResume);
    }

    [Fact]
    public async Task DoubleTalkTooManyRetries_ResumeKeepsCleanSegments()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio,
            _vad,
            _speaker,
            Options(),
            new SpeakerEnrollmentOptions
            {
                CleanSegmentCount = 4,
                MinCleanSegmentsRetained = 4,
                DoubleTalkSegmentCount = 3,
                MinDoubleTalkSegmentsRetained = 2,
                MaxAttemptsPerSegment = 2,
            },
            _stimulus);
        int pass = 0;
        var started = new List<SpeakerEnrollmentProgressEventArgs>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.Kind == SpeakerEnrollmentProgressKind.SegmentStarted)
                lock (started) started.Add(e);
        };
        DriveSegments(coordinator, e => pass == 0 && e.Stage == SpeakerEnrollmentStage.DoubleTalkEnrollment ? 0.002f : 0.85f);

        SpeakerEnrollmentException error = await Assert.ThrowsAsync<SpeakerEnrollmentException>(
            () => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(15)));
        Assert.Equal(SpeakerEnrollmentReasonCode.TooManyRetries, error.Reason);
        Assert.Equal(SpeakerEnrollmentStage.DoubleTalkEnrollment, error.Stage);
        Assert.Equal(0, error.SegmentIndex);
        Assert.True(coordinator.PendingResume.HasValue);
        SpeakerEnrollmentResumeInfo resume = coordinator.PendingResume.Value;
        Assert.Equal(SpeakerEnrollmentStage.DoubleTalkEnrollment, resume.Stage);
        Assert.Equal(0, resume.SegmentIndex);
        Assert.Equal(4, resume.AcceptedCleanSegments);
        Assert.Equal(0, resume.AcceptedDoubleTalkSegments);

        pass = 1;
        started.Clear();
        SpeakerEnrollmentResult result = await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(15));

        Assert.Equal(4, result.CleanSegmentsRetained);
        Assert.Equal(3, result.DoubleTalkSegmentsRetained);
        lock (started)
        {
            Assert.DoesNotContain(started, e => e.Stage == SpeakerEnrollmentStage.CleanEnrollment);
            Assert.Contains(started, e => e.Stage == SpeakerEnrollmentStage.DoubleTalkEnrollment && e.SegmentIndex == 0);
        }
        Assert.Null(coordinator.PendingResume);
    }

    [Fact]
    public async Task DiscardPendingEnrollment_RestartsFromFirstSegment()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio,
            _vad,
            _speaker,
            Options(),
            new SpeakerEnrollmentOptions { CleanSegmentCount = 3, MinCleanSegmentsRetained = 3, DoubleTalkSegmentCount = 0, MaxAttemptsPerSegment = 2 });
        int pass = 0;
        DriveSegments(coordinator, e => pass == 0 && e.SegmentIndex == 2 ? 0.002f : 0.9f);

        await Assert.ThrowsAsync<SpeakerEnrollmentException>(() => coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.True(coordinator.PendingResume.HasValue);
        coordinator.DiscardPendingEnrollment();
        Assert.False(coordinator.PendingResume.HasValue);

        var started = new List<int>();
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.Kind == SpeakerEnrollmentProgressKind.SegmentStarted && e.Stage == SpeakerEnrollmentStage.CleanEnrollment)
                lock (started) started.Add(e.SegmentIndex);
        };
        pass = 1;
        await coordinator.EnrollAsync().WaitAsync(TimeSpan.FromSeconds(10));
        lock (started)
            Assert.Contains(0, started);
    }

    [Fact]
    public async Task Cancellation_StopsAudioAndDetachesSubscriptions()
    {
        var coordinator = new SpeakerEnrollmentCoordinator(
            _audio, _vad, _speaker, Options(), new SpeakerEnrollmentOptions { DoubleTalkSegmentCount = 0 });
        using var cts = new CancellationTokenSource();
        Task<SpeakerEnrollmentResult> enrollment = coordinator.EnrollAsync(cts.Token);
        await Task.Delay(50);
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => enrollment.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(_audio.StartCount, _audio.StopCount);
        Assert.Equal(0, _audio.DuplexSubscriberCount);
        Assert.Equal(0, _vad.SpeakingSubscriberCount);
    }

    private VoiceAssistantSpeakerOptions Options()
        => new() { ProfilePath = Path.Combine(_dir, "owner.json"), Threshold = 0.5f };

    /// <summary>
    /// Feeds one full segment of audio (speech frames with a little silence so the speech ratio is
    /// realistic) every time the coordinator announces a segment attempt.
    /// </summary>
    private void DriveSegments(
        SpeakerEnrollmentCoordinator coordinator,
        Func<SpeakerEnrollmentProgressEventArgs, float> amplitudeFor,
        Func<SpeakerEnrollmentProgressEventArgs, bool>? gapOn = null)
    {
        coordinator.ProgressChanged += (_, e) =>
        {
            if (e.Kind != SpeakerEnrollmentProgressKind.SegmentStarted)
                return;
            Assert.True(e.Attempt >= 1);
            Assert.True(e.Stage is SpeakerEnrollmentStage.CleanEnrollment or SpeakerEnrollmentStage.DoubleTalkEnrollment
                or SpeakerEnrollmentStage.CleanConfirmation or SpeakerEnrollmentStage.DoubleTalkConfirmation);

            float amplitude = amplitudeFor(e);
            bool gap = gapOn?.Invoke(e) ?? false;
            _ = Task.Run(async () =>
            {
                await Task.Delay(10);
                _audio.EmitSegment(amplitude, requiredSpeechSamples: 40000 + 6400, gapAfterFrames: gap ? 12 : -1, _stimulus);
            });
        };
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private sealed class FakeStimulusPlayer : ISpeakerEnrollmentStimulusPlayer
    {
        public bool IsPlaying { get; private set; }
        public bool SilentOnFirstStart { get; set; }
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }

        /// <summary>Whether the current playback actually produces render energy.</summary>
        public bool RenderAudible { get; private set; }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            StartCount++;
            IsPlaying = true;
            RenderAudible = !(SilentOnFirstStart && StartCount == 1);
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            if (IsPlaying)
                StopCount++;
            IsPlaying = false;
            return Task.CompletedTask;
        }
    }

    private sealed class EnrollmentDuplexService : IAudioDuplexService
    {
        private long _sequence;
        private long _position;

        public AudioOptionsBase Options { get; } = new TestOptions();
        public bool IsRecording { get; private set; }
        public bool IsPlaying => false;
        public int CurrentQueueSize => 0;
        public AudioServiceState State { get; private set; } = AudioServiceState.Idle;
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public int SegmentsRecorded { get; private set; }
        public bool EnableAec { get; set; }
        public bool EnableDnr { get; set; }
        public bool EnableHighPassFilter { get; set; }
        public event Action<ReadOnlySpan<float>>? OnDataAvailable;
        public event EventHandler? OnPlaybackCompleted;
        public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;
        public event EventHandler<AudioErrorEventArgs>? OnError;
        public event EventHandler<AudioXrunEventArgs>? OnXrun;
        private DuplexAudioFrameHandler? _duplex;
        public event DuplexAudioFrameHandler? OnDuplexAudioFrame
        {
            add { _duplex += value; }
            remove { _duplex -= value; }
        }
        public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

        public int DuplexSubscriberCount => _duplex?.GetInvocationList().Length ?? 0;

        private readonly object _emitLock = new();

        /// <summary>
        /// Emits one segment's worth of frames. Emitters are serialized: the driver starts one per
        /// announced attempt and a previous emitter may still be flushing its tail, which must stay
        /// contiguous on the shared capture timeline rather than interleave with the next segment.
        /// </summary>
        public void EmitSegment(float amplitude, int requiredSpeechSamples, int gapAfterFrames, FakeStimulusPlayer stimulus)
        {
            lock (_emitLock)
            {
                SegmentsRecorded++;
                int speech = 0;
                int frames = 0;
                while (speech < requiredSpeechSamples && IsRecording)
                {
                    for (int i = 0; i < 4 && speech < requiredSpeechSamples; i++)
                    {
                        EmitFrame(ScriptedSpeakerService.OwnerScoreAudio(amplitude, FrameSamples), stimulus, gap: frames == gapAfterFrames);
                        speech += FrameSamples;
                        frames++;
                    }
                    EmitFrame(new float[FrameSamples], stimulus, gap: frames == gapAfterFrames);
                    frames++;
                }
            }
        }

        private void EmitFrame(float[] capture, FakeStimulusPlayer stimulus, bool gap)
        {
            if (gap)
                _position += FrameSamples; // a dropped frame: the next range does not continue the previous one
            long start = _position;
            long sequence = ++_sequence;
            _position += capture.Length;
            float[] render = new float[capture.Length];
            if (stimulus.IsPlaying && stimulus.RenderAudible)
                Array.Fill(render, 0.1f);
            var frame = new DuplexAudioFrame(
                sequence,
                TimestampTicks: sequence,
                SampleRate: 16000,
                Channels: 1,
                FrameCount: (uint)capture.Length,
                CaptureProcessed: true,
                RenderReference: render,
                CaptureRaw: capture,
                CaptureBeforeAec: capture,
                CaptureAfterAec: capture,
                StreamEpoch: 1,
                StartSample: start);
            _duplex?.Invoke(in frame);
        }

        public Task StartAsync(int inputDeviceIndex = -1, int outputDeviceIndex = -1, CancellationToken cancellationToken = default)
        {
            StartCount++;
            IsRecording = true;
            State = AudioServiceState.Active;
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken = default)
        {
            StopCount++;
            IsRecording = false;
            State = AudioServiceState.Idle;
            return Task.CompletedTask;
        }

        public IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType = null, bool enumAllDevices = false) => [];
        public int GetDefaultInputDevice() => 0;
        public int GetDefaultOutputDevice() => 0;
        public Task StartRecordingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default) => StartAsync(deviceIndex, -1, cancellationToken);
        public Task StopRecordingAsync(CancellationToken cancellationToken = default) => StopAsync(cancellationToken);
        public Task StartPlayingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default) => StartAsync(-1, deviceIndex, cancellationToken);
        public Task StopPlayingAsync(CancellationToken cancellationToken = default) => StopAsync(cancellationToken);
        public Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public void ClearQueue() { }
        public void SetPlaybackGain(float gain) { }
        public bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples) { droppedSamples = 0; return true; }
        public void Dispose() { }

        private sealed class TestOptions : AudioOptionsBase { }
    }

    /// <summary>Energy VAD: every predicted chunk above the floor is reported as speech.</summary>
    private sealed class EnergyVadService : IVoiceActivityDetectService
    {
        private EventHandler<VADEventArgs>? _speaking;
        public event EventHandler<VADEventArgs>? OnSpeechEnded;
        public event EventHandler<EventArgs>? OnSpeechStarted;
        public event EventHandler<VADEventArgs>? OnSpeaking
        {
            add { _speaking += value; }
            remove { _speaking -= value; }
        }
        public event EventHandler<VadProbabilityEventArgs>? OnProbability;

        public int SpeakingSubscriberCount => _speaking?.GetInvocationList().Length ?? 0;

        public void Predict(ReadOnlySpan<float> audioData)
        {
            double energy = 0;
            for (int i = 0; i < audioData.Length; i++)
                energy += audioData[i] * audioData[i];
            if (audioData.Length > 0 && Math.Sqrt(energy / audioData.Length) >= 0.001)
                _speaking?.Invoke(this, new VADEventArgs(audioData.ToArray(), VADEventDataType.Chunk));
        }

        public void Reset() { }
        public void Dispose() { }
    }
}
