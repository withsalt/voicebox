using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ActiveNearEndVadCandidate_RealVerifierControlsPlaybackWhenDtdMissesSpeech(bool owner)
    {
        var services = TestServices.Create();
        string profilePath = await services.WriteOwnerProfileAsync();
        services.Asr.CompleteStart("session-1");
        services.Asr.CompleteStart("session-2");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(
            enableBargeIn: true, enableWakeWord: true, useRealSpeaker: true,
            configure: options =>
            {
                options.Speaker.ProfilePath = profilePath;
                options.Speaker.Confirmation.InferenceTimeoutMs = 2000;
                options.PlaybackPrerollMs = 0;
                options.BargeIn.PlaybackStartupGuardMs = 0;
            });
        try
        {
            await assistant.StartAsync();
            await WakeAsOwnerAsync(services, assistant);
            await SpeakOwnerCommandUntilSpeakingAsync(services, assistant);
            var runtime = GetPrivateField<VoiceBox.Assistant.Runtime.AssistantRuntime>(assistant, "_runtime");
            var nearEnd = RuntimeGraphProbe.RequireField<NearEndDataPlane>(runtime, "_dataPlane");
            services.PlaybackDetector.NearEndSpeechOverride = false;
            services.Speaker.StreamingEmbedding = FakeSpeakerService.UnitEmbedding(owner ? 0 : 1);
            services.Speaker.FeedEntered = new(TaskCreationOptions.RunContinuationsAsynchronously);
            services.Speaker.FeedBlock = new(TaskCreationOptions.RunContinuationsAsynchronously);

            float[] speech = FakeSpeakerService.OwnerAudio(12800);
            services.Audio.Emit(speech);
            await services.Speaker.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking(speech);
            await PumpSilentFramesUntilAsync(services, () => nearEnd.GetDiagnosticSnapshot().CandidateActive, maxFrames: 5);
            Assert.True(nearEnd.GetDiagnosticSnapshot().CandidateActive);
            Assert.Equal(0, services.Audio.ClearQueueCount);
            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));

            // A pending candidate retains its utterance even if VAD ends before identity inference.
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(800));
            services.Speaker.FeedBlock.TrySetResult();
            await PumpSilentFramesUntilAsync(services,
                () => owner ? services.Audio.ClearQueueCount == 1 : nearEnd.CommandDeniedCount == 1);

            Assert.Empty(services.Audio.PlaybackGainChanges);
            Assert.False(services.PlaybackDetector.GetSnapshot().LatestNearEndSpeech);
            if (owner)
            {
                Assert.Equal(1, services.Audio.ClearQueueCount);
                Assert.Equal(1, nearEnd.CommandAuthorizedCount);
                await WaitUntilAsync(() => services.Asr.FedAudio.TryGetValue("session-2", out var chunks) &&
                    chunks.SelectMany(chunk => chunk).Count(sample => Math.Abs(Math.Abs(sample) - 0.3f) < 0.0001f) >= speech.Length);
            }
            else
            {
                Assert.Equal(0, services.Audio.ClearQueueCount);
                Assert.Equal(1, nearEnd.CommandDeniedCount);
                Assert.Equal(0, nearEnd.CommandAuthorizedCount);
                Assert.Equal(AssistantState.Speaking, assistant.State);
                Assert.False(services.Asr.FedAudio.ContainsKey("session-2"));
            }
        }
        finally
        {
            services.Speaker.FeedBlock?.TrySetResult();
            services.Tts.BlockAfterFirstChunk.TrySetResult();
            await assistant.StopAsync();
            File.Delete(profilePath);
        }
    }

    [Fact]
    public void ActiveNearEndVadCandidate_ModelNegativeWaitsForItsOwnSpeakerMatch()
    {
        var speaker = new SequencedSpeakerEvidenceProvider(
            SpeakerVerificationStatus.Pending, SpeakerVerificationStatus.Match);
        var fixture = new VadBargeInFixture(speaker);
        fixture.ObserveVadAudio();

        fixture.Push();

        Assert.True(fixture.Harness.GetDiagnosticSnapshot().CandidateActive);
        Assert.Equal(0, fixture.Confirmations);
        Assert.Equal(0, fixture.Harness.CommandAuthorizedCount);
        SpeakerEvidenceQuery first = Assert.Single(speaker.Queries);
        Assert.Equal(1600, first.CandidateStartSample);
        Assert.False(first.StrongUpstreamEvidence);

        fixture.Push();

        Assert.Equal(first.CandidateId, speaker.Queries[1].CandidateId);
        Assert.Equal(first.CandidateStartSample, speaker.Queries[1].CandidateStartSample);
        Assert.Equal(1, fixture.Confirmations);
        Assert.True(fixture.Harness.TryAcquire(1, out NearEndConfirmationLease lease));
        Assert.True(lease.IdentityAuthorized);
        Assert.Equal(SpeakerVerificationStatus.Match, lease.SpeakerResult.Status);
        Assert.Equal(BargeInAction.Continue, lease.Decision.Action);
        Assert.Equal(0, lease.Decision.AcceptedFrameCount);
        Assert.False(lease.Decision.LatestFrameAccepted);
        Assert.Equal(1600, lease.Selection.SelectedStartSample);
        Assert.Equal(fixture.Audio[1600..], lease.InitialAudio.SelectMany(chunk => chunk).ToArray());
    }

    [Theory]
    [InlineData("Mismatch")]
    [InlineData("InsufficientEvidence")]
    [InlineData("Unavailable")]
    public void ActiveNearEndVadCandidate_RejectsWithoutSpeakerAuthorization(string statusName)
    {
        SpeakerVerificationStatus status = Enum.Parse<SpeakerVerificationStatus>(statusName);
        var speaker = new SequencedSpeakerEvidenceProvider(status);
        var fixture = new VadBargeInFixture(speaker);
        fixture.ObserveVadAudio();

        fixture.Push();

        Assert.Single(speaker.Queries);
        Assert.Equal(0, fixture.Confirmations);
        Assert.Equal(0, fixture.Harness.CommandAuthorizedCount);
        Assert.Equal(1, fixture.Harness.CommandDeniedCount);
        Assert.False(fixture.Harness.GetDiagnosticSnapshot().CandidateActive);
        Assert.Single(speaker.Cancelled);
    }

    [Theory]
    [InlineData("onset-only")]
    [InlineData("unmatched-audio")]
    [InlineData("ended")]
    [InlineData("stale-audio")]
    [InlineData("decision-gap")]
    [InlineData("frame-gap")]
    [InlineData("wrong-epoch")]
    [InlineData("timeline-reset")]
    [InlineData("startup-guard")]
    [InlineData("playback-inactive")]
    [InlineData("not-speaking")]
    public void ActiveNearEndVadCandidate_RequiresCurrentLocatedSpeech(string scenario)
    {
        var speaker = new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Match);
        var fixture = new VadBargeInFixture(speaker);
        fixture.Harness.ObserveEndpointSpeechStarted();
        if (scenario == "unmatched-audio")
            fixture.Harness.ObserveEndpointSpeechAudio(ConstantSamples(0.9f, 320));
        else if (scenario != "onset-only")
            fixture.Harness.ObserveEndpointSpeechAudio(fixture.Audio.AsSpan(1600));

        switch (scenario)
        {
            case "ended":
                fixture.Harness.ObserveUnownedEndpointSpeechEnded();
                break;
            case "stale-audio":
                fixture.Decision = fixture.Decision with { StartSample = 9600, EndSampleExclusive = 15360 };
                break;
            case "decision-gap":
                fixture.Decision = fixture.Decision with { HasDiscontinuity = true };
                break;
            case "frame-gap":
                fixture.Frame = fixture.Frame with { HasDiscontinuity = true };
                break;
            case "wrong-epoch":
                fixture.Decision = fixture.Decision with { StreamEpoch = 2 };
                fixture.Frame = fixture.Frame with { StreamEpoch = 2 };
                break;
            case "timeline-reset":
                ObserveRecognitionAudio(fixture.Harness, 0, new float[320], streamEpoch: 2);
                break;
            case "playback-inactive":
                fixture.PlaybackActive = false;
                break;
            case "not-speaking":
                fixture.Speaking = false;
                break;
        }

        fixture.Harness.ObserveDuplexAudio(fixture.Frame, scenario == "startup-guard");

        Assert.Empty(speaker.Queries);
        Assert.Equal(0, fixture.Confirmations);
        Assert.False(fixture.Harness.GetDiagnosticSnapshot().CandidateActive);
    }

    [Fact]
    public void ActiveNearEndVadCandidate_IdentityDisabledStillRequiresDtdInterrupt()
    {
        var fixture = new VadBargeInFixture(speaker: null);
        fixture.ObserveVadAudio();

        fixture.Push();

        Assert.Equal(0, fixture.Confirmations);
        Assert.False(fixture.Harness.GetDiagnosticSnapshot().CandidateActive);
    }

    [Fact]
    public void ActiveNearEndVadCandidate_DtdOnsetUsesEarlierLocatedVadStart()
    {
        var speaker = new SequencedSpeakerEvidenceProvider(SpeakerVerificationStatus.Pending);
        var fixture = new VadBargeInFixture(speaker);
        fixture.ObserveVadAudio();
        fixture.Decision = fixture.Decision with
        {
            AcceptedFrameCount = 1,
            AcceptedDurationMs = 20,
            LatestFrameAccepted = true,
            FirstAcceptedStartSample = 9280,
        };

        fixture.Push();

        Assert.Equal(1600, Assert.Single(speaker.Queries).CandidateStartSample);
        Assert.Equal(0, fixture.Confirmations);
    }

    private sealed class VadBargeInFixture
    {
        public float[] Audio { get; } = Enumerable.Range(0, 9600).Select(i => 0.03f + i * 0.00001f).ToArray();
        public BargeInDecision Decision = new(BargeInAction.Continue,
            18, 0, 18, 0, 360, 0, 180, 0, 0, 0.0322, 0, 0, 0, "insufficient-nearend-duration",
            StreamEpoch: 1, StartSample: 3840, EndSampleExclusive: 9600);
        public DuplexAudioFrame Frame = CreateDuplexFrame(1, 9280, 320);
        public bool PlaybackActive = true;
        public bool Speaking = true;
        public int Confirmations;
        public NearEndTestHarness Harness { get; }

        public VadBargeInFixture(ISpeakerEvidenceProvider? speaker)
        {
            var services = TestServices.Create();
            Harness = CreateNearEndTestHarness(new VoiceAssistantBargeInOptions
            {
                Enable = true,
                ActiveNearEndPreRollMs = 0,
            }, new TestRecognitionAudioPort(services.Asr), () => Speaking, (_, _, _) => Decision,
                new AssistantAudioFormat(new DuplexAssistantAudioFrontend(services.Audio).Input,
                    new SpeechSynthesisStreamAdapter(services.Tts, null, null)), NullLogger.Instance,
                () => false, () => PlaybackActive, _ => { Confirmations++; return true; },
                (_, _) => { }, speaker);
            ObserveRecognitionAudio(Harness, 0, Audio);
        }

        public void ObserveVadAudio()
        {
            Harness.ObserveEndpointSpeechStarted();
            Harness.ObserveEndpointSpeechAudio(Audio.AsSpan(1600));
        }

        public void Push() => Harness.ObserveDuplexAudio(Frame, playbackStartupGuardActive: false);
    }
}
