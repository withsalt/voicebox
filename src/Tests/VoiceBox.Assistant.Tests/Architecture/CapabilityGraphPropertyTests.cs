using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Core.Services.AudioService;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(11)]
    [InlineData(17)]
    [InlineData(23)]
    [InlineData(29)]
    [InlineData(31)]
    [InlineData(42)]
    [InlineData(99)]
    [InlineData(128)]
    public void CapabilityGraph_RandomLegalCombinationsBuildBeforeAnyProviderStart(int seed)
    {
        var rng = new Random(seed);
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = CreateLegalBuilder(services, rng);
        AssistantComposition composition = builder.BuildComposition();
        Assert.Contains("SpeechRecognition", composition.Snapshot.Capabilities);
        Assert.Equal(0, services.Audio.StartCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(13)]
    [InlineData(21)]
    [InlineData(34)]
    [InlineData(55)]
    [InlineData(89)]
    [InlineData(144)]
    public void CapabilityGraph_RandomIllegalCombinationsFailBeforeStartOrSubscribe(int seed)
    {
        var rng = new Random(seed);
        var services = TestServices.Create();
        int kind = rng.Next(7);
        VoiceAssistantBuilder builder;
        string expectedCode;
        switch (kind)
        {
            case 6:
                // Confirmation.Enable without a speaker provider is never legal.
                builder = CreateMinimalLegalBuilder(services)
                    .UseOptions(new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } })
                    .AddWakeup(services.Wakeup);
                expectedCode = "VBX-COMP-004";
                break;
            case 0:
                builder = VoiceAssistant.CreateBuilder()
                    .UseTurnBoundary(services.Vad)
                    .UseSpeechRecognition(services.Asr)
                    .UseResponseGeneration(services.Llm)
                    .UseSpeechSynthesis(services.Tts);
                expectedCode = "VBX-COMP-001";
                break;
            case 1:
                builder = VoiceAssistant.CreateBuilder()
                    .UseAudio(services.Audio)
                    .UseSpeechRecognition(services.Asr)
                    .UseResponseGeneration(services.Llm)
                    .UseSpeechSynthesis(services.Tts);
                expectedCode = "VBX-COMP-001";
                break;
            case 2:
                builder = VoiceAssistant.CreateBuilder()
                    .UseAudio(services.Audio)
                    .UseTurnBoundary(services.Vad)
                    .UseResponseGeneration(services.Llm)
                    .UseSpeechSynthesis(services.Tts);
                expectedCode = "VBX-COMP-001";
                break;
            case 3:
                builder = CreateMinimalLegalBuilder(services);
                AssistantCompositionException duplicate = Assert.Throws<AssistantCompositionException>(
                    () => builder.UseSpeechRecognition(services.Asr));
                Assert.Equal("VBX-COMP-002", duplicate.Code);
                Assert.Equal(0, services.Audio.StartCount);
                return;
            case 4:
                var record = new FakeDuplexService();
                var playback = new FakeDuplexService();
                builder = VoiceAssistant.CreateBuilder()
                    .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
                    .UseTurnBoundary(services.Vad)
                    .UseSpeechRecognition(services.Asr)
                    .UseResponseGeneration(services.Llm)
                    .UseSpeechSynthesis(services.Tts)
                    .AddBargeIn(
                        new VoiceAssistantBargeInOptions(),
                        TestAcousticPathDelayEstimatorFactory.Instance);
                expectedCode = "VBX-COMP-008";
                AssistantCompositionException split =
                    Assert.Throws<AssistantCompositionException>(builder.Build);
                Assert.Equal(expectedCode, split.Code);
                Assert.Equal(0, record.StartCount);
                Assert.Equal(0, playback.StartCount);
                return;
            default:
                builder = CreateMinimalLegalBuilder(services);
                builder.RequireCapability(AssistantCapability.StructuredToolUse);
                expectedCode = "VBX-COMP-011";
                break;
        }

        AssistantCompositionException failure =
            Assert.Throws<AssistantCompositionException>(builder.Build);
        Assert.Equal(expectedCode, failure.Code);
        Assert.Equal(0, services.Audio.StartCount);
    }

    private static VoiceAssistantBuilder CreateMinimalLegalBuilder(TestServices services)
        => VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);

    private static VoiceAssistantBuilder CreateLegalBuilder(TestServices services, Random rng)
    {
        VoiceAssistantBuilder builder = CreateMinimalLegalBuilder(services);
        bool wakeup = rng.Next(2) == 1;
        if (wakeup)
            builder.AddWakeup(services.Wakeup);
        // Confirmation defaults to off, so a speaker provider is optional and installs no online
        // identity module. KWS without a provider is legal.
        if (wakeup || rng.Next(2) == 1)
            builder.AddSpeakerVerification(services.Speaker);
        if (rng.Next(2) == 1)
            builder.AddBargeIn(
                new VoiceAssistantBargeInOptions(),
                TestAcousticPathDelayEstimatorFactory.Instance);
        return builder;
    }
}
