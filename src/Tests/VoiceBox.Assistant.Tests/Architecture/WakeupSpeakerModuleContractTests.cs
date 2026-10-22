using System.Reflection;
using System.Threading.Channels;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class WakeupSpeakerModuleContractTests
{
    [Fact]
    public void RuntimeDependsOnOptionalModulePorts_NotConcreteWakeupOrSpeakerImplementations()
    {
        FieldInfo[] fields = ShellSurface.Fields;

        Assert.DoesNotContain(fields, field =>
            field.FieldType == typeof(WakeupSessionGate) ||
            field.FieldType == typeof(SpeechGatedSpeakerVerifier));
        Assert.Contains(fields, field => field.FieldType == typeof(IWakeupModule));
        Assert.Contains(fields, field => field.FieldType == typeof(ISpeakerModule));
    }

    [Fact]
    public void ModulePortsAcceptOnlyTypedCaptureTimelineData_AndExposeNoRuntimeLookup()
    {
        Type[] moduleTypes = [typeof(WakeupModule), typeof(SpeakerModule)];
        foreach (Type module in moduleTypes)
        {
            Assert.DoesNotContain(
                module.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
                field => field.FieldType == typeof(AssistantRuntime) ||
                    field.Name.Contains("serviceProvider", StringComparison.OrdinalIgnoreCase) ||
                    field.Name.Contains("locator", StringComparison.OrdinalIgnoreCase));
        }

        Assert.Contains(
            typeof(WakeupModule).GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic),
            method => method.GetParameters().Any(parameter =>
                parameter.ParameterType == typeof(ProcessedCaptureFrame).MakeByRefType()));
        Assert.Contains(
            typeof(SpeakerModule).GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic),
            method => method.GetParameters().Any(parameter =>
                parameter.ParameterType == typeof(SpeakerCaptureFrame).MakeByRefType()));
    }

    /// <summary>
    /// The speaker module consumes the identity tap only. It must never accept the final processed
    /// capture frame again: that path runs after the independent DNR and is not what enrollment saw.
    /// </summary>
    [Fact]
    public void SpeakerModule_ConsumesIdentityTapNotProcessedCapture()
    {
        Assert.True(typeof(ISpeakerCaptureFrameSink).IsAssignableFrom(typeof(ISpeakerModule)));
        Assert.False(typeof(IProcessedCaptureFrameSink).IsAssignableFrom(typeof(SpeakerModule)));
        Assert.DoesNotContain(
            typeof(SpeakerModule).GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic),
            method => method.GetParameters().Any(parameter =>
                parameter.ParameterType == typeof(ProcessedCaptureFrame).MakeByRefType() ||
                parameter.ParameterType == typeof(ProcessedCaptureFrame)));
        Assert.Null(typeof(VoiceAssistant).Assembly.GetType("VoiceBox.Assistant.Modules.Speaker.ContinuousSpeakerTracker"));
        Assert.Null(typeof(VoiceAssistant).Assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeakerAudioQueue"));
    }

    [Fact]
    public void LifecycleParticipantNamesRemainIndependentAndOptional()
    {
        Assert.Equal(
            ["AudioFrontend", "WakeupGate", "SpeakerEvidence", "BargeIn", "ResponseGeneration"],
            Enum.GetNames<AssistantLifecycleParticipant>());

        Assert.NotEqual(typeof(IWakeupModule), typeof(ISpeakerModule));
        Assert.True(typeof(IWakeupModule).IsInterface);
        Assert.True(typeof(ISpeakerModule).IsInterface);
        Assert.True(typeof(ISpeakerEvidenceProvider).IsAssignableFrom(typeof(ISpeakerModule)));
        Assert.Equal(
            typeof(AssistantTimelineRequirement),
            typeof(IWakeupModule).GetProperty(nameof(IWakeupModule.TimelineRequirement))!.PropertyType);
        Assert.Equal(
            typeof(AssistantTimelineRequirement),
            typeof(ISpeakerModule).GetProperty(nameof(ISpeakerModule.TimelineRequirement))!.PropertyType);
    }

    [Fact]
    public void SpeakerModuleKeepsEvidencePrivateAndOwnsOneBoundedScheduler()
    {
        Assert.Contains(
            typeof(SpeakerModule).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == typeof(SpeechGatedSpeakerVerifier));
        Assert.DoesNotContain(
            typeof(SpeakerModule).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == typeof(Channel<>) ||
                field.FieldType.Name.Contains("Dictionary", StringComparison.Ordinal));
        Assert.Contains(typeof(ISpeakerCaptureFrameSink).GetMethods(), method => method.Name == "OnSpeakerCapture");
        Assert.Contains(typeof(ISpeakerModule).GetMethods(), method => method.Name == "StopAsync");
        Assert.Contains(typeof(ISpeakerEvidenceProvider).GetMethods(), method => method.Name == "Cancel");
        // Speaker evidence is optional: Confirmation.Enable off keeps a keyword-only gate.
        System.Reflection.ParameterInfo speakerParameter = Assert.Single(
            typeof(IWakeupModuleFactory).GetMethod(nameof(IWakeupModuleFactory.Create))!.GetParameters(),
            parameter => parameter.ParameterType == typeof(ISpeakerEvidenceProvider));
        Assert.False(speakerParameter.HasDefaultValue);
        Assert.Contains(
            speakerParameter.GetCustomAttributesData(),
            static attribute => attribute.AttributeType.Name.Contains("Nullable", StringComparison.Ordinal) &&
                attribute.ConstructorArguments.Any(static argument => argument.Value is byte and 2));
        Assert.Contains(
            typeof(WakeupSessionGate).GetConstructors().Single().GetParameters(),
            parameter => parameter.ParameterType == typeof(ISpeakerEvidenceProvider) && !parameter.HasDefaultValue);
    }
}

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void OptionalWakeupAndSpeakerModules_FormIndependentCompositionVariants()
    {
        var services = TestServices.Create();

        var confirm = new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } };
        AssistantComposition wakeupWithSpeaker = VoiceAssistant.CreateBuilder()
            .UseOptions(confirm)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();
        Assert.Equal(
            [
                AssistantLifecycleParticipant.SpeakerEvidence,
                AssistantLifecycleParticipant.WakeupGate,
                AssistantLifecycleParticipant.AudioFrontend,
            ],
            wakeupWithSpeaker.Lifecycle.StartOrder);
        Assert.Contains("WakeupGate", wakeupWithSpeaker.Snapshot.Capabilities);
        Assert.Contains("SpeakerEvidence", wakeupWithSpeaker.Snapshot.Capabilities);
        Assert.Contains("SpeakerCaptureTimeline", wakeupWithSpeaker.Snapshot.Capabilities);
        Assert.Contains("Speaker:Requires:SpeakerCaptureTimeline:0", wakeupWithSpeaker.Snapshot.CapabilityEdges);
        Assert.DoesNotContain("Speaker:Requires:ProcessedCaptureTimeline:0", wakeupWithSpeaker.Snapshot.CapabilityEdges);
        Assert.Contains("Wakeup:Requires:SpeakerEvidence:0", wakeupWithSpeaker.Snapshot.CapabilityEdges);

        // Confirmation off: an installed speaker provider is ignored for online verification.
        AssistantComposition speakerWithoutKws = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();
        Assert.Equal([AssistantLifecycleParticipant.AudioFrontend], speakerWithoutKws.Lifecycle.StartOrder);
        Assert.DoesNotContain("WakeupGate", speakerWithoutKws.Snapshot.Capabilities);
        Assert.DoesNotContain("SpeakerEvidence", speakerWithoutKws.Snapshot.Capabilities);
        Assert.Null(speakerWithoutKws.RuntimePlan.Speaker);
        Assert.DoesNotContain(speakerWithoutKws.Snapshot.Ownership, ownership => ownership.ModuleId.StartsWith("Speaker.", StringComparison.Ordinal));

        AssistantComposition absent = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();
        Assert.Equal([AssistantLifecycleParticipant.AudioFrontend], absent.Lifecycle.StartOrder);
        Assert.DoesNotContain("WakeupGate", absent.Snapshot.Capabilities);
        Assert.DoesNotContain("SpeakerEvidence", absent.Snapshot.Capabilities);
    }

    [Fact]
    public void KwsEnabled_WithoutSpeakerProvider_FailsComposition()
    {
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } })
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup);

        AssistantCompositionException exception =
            Assert.Throws<AssistantCompositionException>(builder.BuildComposition);
        Assert.Equal("VBX-COMP-004", exception.Code);
    }

    [Fact]
    public void KwsEnabled_DerivesWakeupAndBargeInPurposesFromSharedConfirmation()
    {
        var services = TestServices.Create();
        var confirm = new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } };
        AssistantComposition wakeupOnly = VoiceAssistant.CreateBuilder()
            .UseOptions(confirm)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();
        Assert.Contains("SpeakerEvidence", wakeupOnly.Snapshot.Capabilities);
        Assert.Contains("WakeupGate", wakeupOnly.Snapshot.Capabilities);
        Assert.NotNull(wakeupOnly.RuntimePlan.Speaker);

        AssistantComposition withBargeIn = VoiceAssistant.CreateBuilder()
            .UseOptions(confirm)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .AddBargeIn(new VoiceAssistantBargeInOptions(), TestAcousticPathDelayEstimatorFactory.Instance)
            .BuildComposition();
        Assert.Contains("SpeakerEvidence", withBargeIn.Snapshot.Capabilities);
        Assert.Contains("BargeIn", withBargeIn.Snapshot.Capabilities);
    }

    [Fact]
    public void OptionalWakeupModule_RejectsSplitAudioWithoutIdentityTap()
    {
        var services = TestServices.Create();
        var record = new VoiceAssistantLifecycleTests.FakeDuplexService();
        var playback = new VoiceAssistantLifecycleTests.FakeDuplexService();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } })
            .UseAudio((VoiceBox.Core.Services.AudioService.IAudioRecordService)record,
                (VoiceBox.Core.Services.AudioService.IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker);

        AssistantCompositionException exception =
            Assert.Throws<AssistantCompositionException>(builder.BuildComposition);
        Assert.Equal("VBX-COMP-013", exception.Code);

        // Without KWS a split frontend with an installed speaker provider composes normally.
        AssistantComposition speakerWithoutKws = VoiceAssistant.CreateBuilder()
            .UseAudio((VoiceBox.Core.Services.AudioService.IAudioRecordService)record,
                (VoiceBox.Core.Services.AudioService.IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();
        Assert.Null(speakerWithoutKws.RuntimePlan.Speaker);
    }

    [Fact]
    public void SingleTalkEnabled_RequiresProcessedCaptureTimelineOnTurnBoundary()
    {
        var services = TestServices.Create();
        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseOptions(new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } })
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();

        Assert.Contains(
            "TurnBoundary:Requires:ProcessedCaptureTimeline:0",
            composition.Snapshot.CapabilityEdges);
        Assert.DoesNotContain(
            "Speaker:Requires:ProcessedCaptureTimeline:0",
            composition.Snapshot.CapabilityEdges);
    }
}
