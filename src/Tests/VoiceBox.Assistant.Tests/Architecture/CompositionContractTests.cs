using System.Reflection;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using Xunit;
using KernelEffectKind = VoiceBox.Assistant.Kernel.Effects.AssistantEffectKind;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Contracts for the typed composition root: capability validation, lifecycle plan generation and
/// deterministic build failures. Reflection is used where the assertion is about the shape of the
/// composition surface itself rather than about runtime behaviour.
/// </summary>
public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void TypedCompositionRoot_IsAnExplicitAdditivePublicContract()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;

        Type builder = Require(assembly.GetType("VoiceBox.Assistant.Composition.Builder.VoiceAssistantBuilder"));
        MethodInfo createBuilder = Require(typeof(VoiceAssistant).GetMethod(
            "CreateBuilder",
            BindingFlags.Public | BindingFlags.Static));

        Assert.Equal(builder, createBuilder.ReturnType);
    }

    [Fact]
    public void TypedCompositionSnapshot_ContainsDeclaredCapabilitiesAndStaticEffectBindings()
    {
        var services = TestServices.Create();

        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();

        Assert.Equal("BasicDuplexQa", composition.Snapshot.ProfileId);
        Assert.Equal(
        [
            "AudioCapture",
            "AudioPlayback",
            "CaptureTimeline",
            "TurnBoundary",
            "SpeechRecognition",
            "ResponseGeneration",
            "SpeechSynthesis",
            "SimultaneousCapturePlayback",
            "RenderReference",
            "ProcessedCaptureTimeline",
            "CaptureProcessorStage",
            "SpeakerCaptureTimeline",
        ],
        composition.Snapshot.Capabilities);
        Assert.Equal(Enum.GetValues<KernelEffectKind>().Length, composition.Wiring.EffectBindings.Count);
        Assert.All(
            Enum.GetValues<KernelEffectKind>(),
            effect => Assert.Single(composition.Wiring.EffectBindings, binding => binding.Effect == effect));
        Assert.Equal([AssistantLifecycleParticipant.AudioFrontend], composition.Lifecycle.StartOrder);
    }

    [Fact]
    public void OptionalModules_AppearOnlyAfterExplicitTypedRegistration()
    {
        var services = TestServices.Create();
        VoiceAssistantOptions options = new()
        {
            Wakeup = { Enable = true },
            Speaker = { Confirmation = { Enable = true } },
        };
        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .UseOptions(options)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .BuildComposition();

        Assert.Contains("WakeupGate", composition.Snapshot.Capabilities);
        Assert.Contains("SpeakerEvidence", composition.Snapshot.Capabilities);
        // Identity evidence and the wakeup gate are running before the frontend emits its first frame.
        Assert.Equal(
        [
            AssistantLifecycleParticipant.SpeakerEvidence,
            AssistantLifecycleParticipant.WakeupGate,
            AssistantLifecycleParticipant.AudioFrontend,
        ],
        composition.Lifecycle.StartOrder);
    }

    [Fact]
    public void BuilderContract_UsesTypedLeafRolesOnly()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type builder = Require(assembly.GetType("VoiceBox.Assistant.Composition.Builder.VoiceAssistantBuilder"));

        AssertBuilderMethod(builder, "UseAudio", typeof(IAudioDuplexService));
        AssertBuilderMethod(builder, "UseAudio", typeof(IAudioRecordService), typeof(IAudioPlayService));
        AssertBuilderMethod(builder, "UseTurnBoundary", typeof(IVoiceActivityDetectService));
        AssertBuilderMethod(builder, "UseSpeechRecognition", typeof(IASRService));
        AssertBuilderMethod(builder, "UseResponseGeneration", typeof(IStreamLLMService));
        AssertBuilderMethod(builder, "UseSpeechSynthesis", typeof(IStreamTTSService));

        Assert.DoesNotContain(builder.GetMethods(BindingFlags.Public | BindingFlags.Instance),
            static method => method.GetParameters().Any(static parameter =>
                parameter.ParameterType.Name is "IServiceProvider" or "IAudioServiceManager"));
    }

    [Fact]
    public async Task BasicDuplexQa_BuildsAndRunsThroughTheTypedPlan()
    {
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);

        AssistantComposition composition = builder.BuildComposition();
        Assert.Equal("BasicDuplexQa", composition.Snapshot.ProfileId);
        Assert.Contains("SimultaneousCapturePlayback", composition.Snapshot.Capabilities);
        Assert.Contains("RenderReference", composition.Snapshot.Capabilities);
        Assert.Contains("ProcessedCaptureTimeline", composition.Snapshot.Capabilities);

        using VoiceAssistant assistant = builder.Build();
        await assistant.StartAsync();
        await assistant.StopAsync();

        Assert.Equal(1, services.Audio.StartCount);
        Assert.Equal(1, services.Audio.StopCount);
    }

    [Fact]
    public async Task BasicSplitAudioQa_BuildsWithoutRenderCapabilitiesAndStartsEachEndpointOnce()
    {
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);

        AssistantComposition composition = builder.BuildComposition();
        Assert.Equal("BasicSplitAudioQa", composition.Snapshot.ProfileId);
        Assert.DoesNotContain("SimultaneousCapturePlayback", composition.Snapshot.Capabilities);
        Assert.DoesNotContain("RenderReference", composition.Snapshot.Capabilities);
        Assert.DoesNotContain("ProcessedCaptureTimeline", composition.Snapshot.Capabilities);

        using VoiceAssistant assistant = builder.Build();
        await assistant.StartAsync();
        await assistant.StopAsync();

        Assert.Equal(1, record.StartCount);
        Assert.Equal(1, record.StopCount);
        Assert.Equal(1, playback.StartCount);
        Assert.Equal(1, playback.StopCount);
    }

    [Fact]
    public void SharedDuplexPassedAsRecordAndPlay_IsDeduplicatedBeforeTheLifecyclePlan()
    {
        var services = TestServices.Create();
        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseAudio((IAudioRecordService)services.Audio, (IAudioPlayService)services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();

        Assert.Equal("BasicDuplexQa", composition.Snapshot.ProfileId);
        Assert.Single(composition.Lifecycle.StartOrder);
        Assert.Equal(AssistantLifecycleParticipant.AudioFrontend, composition.Lifecycle.StartOrder[0]);
    }

    [Fact]
    public void MissingRoleDuplicateRoleAndUnsupportedBargeIn_FailBeforeRuntimeSideEffects()
    {
        var services = TestServices.Create();
        VoiceAssistantBuilder incomplete = VoiceAssistant.CreateBuilder().UseAudio(services.Audio);
        AssistantCompositionException missing =
            Assert.Throws<AssistantCompositionException>(incomplete.Build);
        Assert.Equal("VBX-COMP-001", missing.Code);
        Assert.Equal(0, services.Audio.StartCount);

        AssistantCompositionException duplicate = Assert.Throws<AssistantCompositionException>(
            () => VoiceAssistant.CreateBuilder().UseAudio(services.Audio).UseAudio(services.Audio));
        Assert.Equal("VBX-COMP-002", duplicate.Code);
        Assert.Equal(0, services.Audio.StartCount);

        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        VoiceAssistantBuilder splitBargeIn = VoiceAssistant.CreateBuilder()
            .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddBargeIn(
                new VoiceAssistantBargeInOptions(),
                TestAcousticPathDelayEstimatorFactory.Instance);
        AssistantCompositionException incompatible =
            Assert.Throws<AssistantCompositionException>(splitBargeIn.Build);
        Assert.Equal("VBX-COMP-008", incompatible.Code);
        Assert.Equal(0, record.StartCount);
        Assert.Equal(0, playback.StartCount);
    }

    [Fact]
    public void RuntimeConstructor_ConsumesOnlyImmutableCompositionPlans()
    {
        ConstructorInfo[] constructors = typeof(AssistantRuntime).GetConstructors(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

        // The facade assembles nothing: Composition hands it one finished collaborator set,
        // which itself was built from the immutable plans.
        Assert.All(constructors, static constructor =>
        {
            Type[] parameters = constructor.GetParameters().Select(static parameter => parameter.ParameterType).ToArray();
            Assert.Contains(typeof(AssistantRuntimeCompositionServices), parameters);
            Assert.DoesNotContain(typeof(AssistantComposition), parameters);
            Assert.DoesNotContain(typeof(AssistantCapabilityGraph), parameters);
            Assert.DoesNotContain(parameters, static type => type.Name is "IAudioServiceManager" or "IServiceProvider");
        });

        MethodInfo compositionFactory = typeof(AssistantRuntimeCompositionFactory).GetMethod(
            "Create",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!;
        Type[] factoryParameters = [.. compositionFactory.GetParameters().Select(static p => p.ParameterType)];
        Assert.Contains(typeof(AssistantRuntimePlan), factoryParameters);
        Assert.Contains(typeof(ModuleLifecyclePlan), factoryParameters);
        Assert.DoesNotContain(typeof(ModuleWiring), factoryParameters);
        Assert.DoesNotContain(typeof(AssistantCapabilityGraph), factoryParameters);
        Assert.DoesNotContain(
            typeof(AssistantRuntime).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            static field => field.FieldType == typeof(AssistantCapabilityGraph));
    }

    [Fact]
    public async Task SplitAudioFrontend_ExposesTypedPortsAndDoesNotManufactureRenderFrames()
    {
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        using var frontend = new SplitAssistantAudioFrontend(record, playback);
        int processedCallbacks = 0;
        int processedSamples = 0;
        Assert.Null(frontend.RenderReference);
        Assert.NotNull(frontend.ProcessedCapture);
        frontend.ProcessedCapture!.OnProcessedCaptureFrame += (in ProcessedCaptureFrame frame) =>
        {
            processedCallbacks++;
            processedSamples += frame.SampleCount;
            Assert.True(frame.SourceRange.IsValid);
        };

        await frontend.StartAsync();
        Assert.True(frontend.Input.IsCapturing);
        Assert.Equal(AudioServiceState.Active, frontend.Output.State);
        Assert.Equal(record.Options.SampleRate, frontend.Input.SampleRate);
        Assert.Equal(record.Options.ChannelCount, frontend.Input.ChannelCount);

        record.EmitRecordData([0.1f, 0.2f, 0.3f]);
        Assert.Equal(1, processedCallbacks);
        Assert.Equal(3, processedSamples);

        record.EmitRecordData([]);
        Assert.Equal(1, processedCallbacks);

        Assert.True(frontend.Output.TryEnqueue([0.4f, 0.5f], out int dropped));
        Assert.Equal(0, dropped);
        frontend.Output.SetGain(0.3f);
        Assert.Equal(0.3f, playback.PlaybackGain);
        frontend.Output.ClearQueue();
        await frontend.StopAsync();

        frontend.Dispose();
        record.EmitRecordData([0.6f]);
        Assert.Equal(1, processedCallbacks);
    }

    [Fact]
    public async Task SplitAudioAdapter_StartsPlaybackBeforeCapture_RollsBackPartialStart_AndPreservesTheFirstStopFailure()
    {
        var orderedRecord = new FakeDuplexService
        {
            StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StopEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StopBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        var orderedPlayback = new FakeDuplexService
        {
            StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StopEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StopBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        using var orderedAdapter = new SplitAssistantAudioFrontend(orderedRecord, orderedPlayback);

        Task orderedStart = orderedAdapter.StartAsync();
        await orderedPlayback.StartEntered!.Task.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal(0, orderedRecord.StartCount);
        orderedPlayback.StartBlock!.TrySetResult();
        await orderedRecord.StartEntered!.Task.WaitAsync(TimeSpan.FromSeconds(1));
        await orderedStart;

        Task orderedStop = orderedAdapter.StopAsync();
        await orderedRecord.StopEntered!.Task.WaitAsync(TimeSpan.FromSeconds(1));
        Assert.Equal(0, orderedPlayback.StopCount);
        orderedRecord.StopBlock!.TrySetResult();
        await orderedPlayback.StopEntered!.Task.WaitAsync(TimeSpan.FromSeconds(1));
        orderedPlayback.StopBlock!.TrySetResult();
        await orderedStop;

        var record = new FakeDuplexService
        {
            StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
            StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously),
        };
        var playback = new FakeDuplexService();
        using var adapter = new SplitAssistantAudioFrontend(record, playback);
        using var cancellation = new CancellationTokenSource();

        Task partialStart = adapter.StartAsync(cancellation.Token);
        await record.StartEntered!.Task.WaitAsync(TimeSpan.FromSeconds(1));
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => partialStart);
        Assert.Equal(1, record.StartCount);
        Assert.Equal(0, record.StopCount);
        Assert.Equal(1, playback.StartCount);
        Assert.Equal(1, playback.StopCount);

        var stopRecord = new FakeDuplexService();
        var stopPlayback = new FakeDuplexService
        {
            StopException = new InvalidOperationException("playback stop"),
        };
        using var stopAdapter = new SplitAssistantAudioFrontend(stopRecord, stopPlayback);
        InvalidOperationException playbackFailure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => stopAdapter.StopAsync());
        Assert.Equal("playback stop", playbackFailure.Message);
        Assert.Equal(1, stopRecord.StopCount);

        var recordFailure = new FakeDuplexService
        {
            StopException = new InvalidOperationException("record stop"),
        };
        var successfulPlayback = new FakeDuplexService();
        using var recordFailureAdapter = new SplitAssistantAudioFrontend(recordFailure, successfulPlayback);
        InvalidOperationException capturedRecordFailure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => recordFailureAdapter.StopAsync());
        Assert.Equal("record stop", capturedRecordFailure.Message);
        Assert.Equal(1, successfulPlayback.StopCount);

        var bothFailureRecord = new FakeDuplexService
        {
            StopException = new InvalidOperationException("record stop"),
        };
        var bothFailurePlayback = new FakeDuplexService
        {
            StopException = new InvalidOperationException("playback stop"),
        };
        using var bothFailureAdapter = new SplitAssistantAudioFrontend(bothFailureRecord, bothFailurePlayback);
        InvalidOperationException firstFailure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => bothFailureAdapter.StopAsync());
        Assert.Equal("record stop", firstFailure.Message);
    }

    [Fact]
    public void CapabilityValidator_RejectsFormatAndOptionalDependencyViolationsBeforeStart()
    {
        var services = TestServices.Create();
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        playback.Options.SampleRate = 48000;
        VoiceAssistantBuilder incompatibleAudio = VoiceAssistant.CreateBuilder()
            .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);
        AssistantCompositionException format =
            Assert.Throws<AssistantCompositionException>(incompatibleAudio.Build);
        Assert.Equal("VBX-COMP-006", format.Code);

        VoiceAssistantBuilder builderIgnoringWakeupEnableFlag = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .UseOptions(new VoiceAssistantOptions { Wakeup = { Enable = true } });
        using VoiceAssistant assistantWithoutWakeup = builderIgnoringWakeupEnableFlag.Build();

        // Confirmation.Enable requires the owner-speaker provider; KWS alone does not.
        VoiceAssistantBuilder builderWithWakeupOnly = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .UseOptions(new VoiceAssistantOptions
            {
                Speaker = { Confirmation = { Enable = true } },
            })
            .AddWakeup(services.Wakeup);
        AssistantCompositionException keywordOnly =
            Assert.Throws<AssistantCompositionException>(builderWithWakeupOnly.Build);
        Assert.Equal("VBX-COMP-004", keywordOnly.Code);
        Assert.Equal(0, services.Audio.StartCount);
        Assert.Equal(0, record.StartCount);
        Assert.Equal(0, playback.StartCount);
    }

    [Fact]
    public void AudioFrontendCapabilities_AreStaticAndMatchTheSelectedTransport()
    {
        var duplex = new FakeDuplexService();
        var splitRecord = new FakeDuplexService();
        var splitPlayback = new FakeDuplexService();
        var duplexFrontend = new DuplexAssistantAudioFrontend(duplex);
        using var splitFrontend = new SplitAssistantAudioFrontend(splitRecord, splitPlayback);

        // The typed ports are the only audio surface; no provider instance is reachable.
        Assert.Equal(duplex.Options.SampleRate, duplexFrontend.Input.SampleRate);
        Assert.Equal(duplex.Options.ChannelCount, duplexFrontend.Output.ChannelCount);
        Assert.NotNull(duplexFrontend.ProcessedCapture);
        Assert.NotNull(duplexFrontend.RenderReference);
        Assert.True(duplexFrontend.SupportsSimultaneousCapturePlayback);
        Assert.True(duplexFrontend.SupportsRenderReference);
        Assert.True(duplexFrontend.SupportsProcessedCaptureTimeline);

        Assert.Equal(splitRecord.Options.SampleRate, splitFrontend.Input.SampleRate);
        splitFrontend.Output.SetGain(0.5f);
        Assert.Equal(0.5f, splitPlayback.PlaybackGain);
        Assert.Null(splitFrontend.RenderReference);
        Assert.False(splitFrontend.SupportsSimultaneousCapturePlayback);
        Assert.False(splitFrontend.SupportsRenderReference);
        Assert.False(splitFrontend.SupportsProcessedCaptureTimeline);

        Assert.Null(typeof(VoiceAssistant).Assembly.GetType(
            "VoiceBox.Assistant.Audio.SplitAudioDuplexAdapter"));
        Assert.DoesNotContain(
            typeof(IAssistantAudioFrontend).GetProperties(),
            property => property.PropertyType == typeof(IAudioDuplexService) ||
                property.PropertyType == typeof(IAudioRecordService) ||
                property.PropertyType == typeof(IAudioPlayService));
    }

    [Fact]
    public void CapabilityValidator_RejectsEveryZeroAudioFormatFieldBeforeRuntimeStartup()
    {
        AssertZeroAudioFormatIsRejected(static (record, _) => record.Options.SampleRate = 0);
        AssertZeroAudioFormatIsRejected(static (record, _) => record.Options.ChannelCount = 0);
        AssertZeroAudioFormatIsRejected(static (_, playback) => playback.Options.SampleRate = 0);
        AssertZeroAudioFormatIsRejected(static (_, playback) => playback.Options.ChannelCount = 0);
    }

    [Fact]
    public void CapabilityValidator_RequiresEveryBargeInAudioCapability()
    {
        AssertBargeInCapabilityRejected(supportsSimultaneousCapturePlayback: false, true, true);
        AssertBargeInCapabilityRejected(true, supportsRenderReference: false, true);
        AssertBargeInCapabilityRejected(true, true, supportsProcessedCaptureTimeline: false);
    }

    [Fact]
    public void CapabilityValidator_SingleTalkRequiresLocatableProcessedSpeechTimeline()
    {
        var services = TestServices.Create();
        var audio = new CapabilityTestAudioFrontend(
            new FakeDuplexService(),
            supportsSimultaneousCapturePlayback: true,
            supportsRenderReference: true,
            supportsProcessedCaptureTimeline: false);
        var graph = new AssistantCapabilityGraph(
            AssistantCompositionProfile.BasicDuplexQa,
            audio,
            services.Vad,
            services.Asr,
            services.Llm,
            services.Tts,
            wakeup: services.Wakeup,
            speaker: services.Speaker,
            acousticDiagnostics: null,
            new VoiceAssistantOptions
            {
                Wakeup = { Enable = true },
                Speaker = { Confirmation = { Enable = true } },
            });

        AssertCompositionCode("VBX-COMP-015", () => graph.ValidateAndCreateComposition());
        Assert.Equal(0, audio.Runtime.StartCount);
    }

    [Fact]
    public void BuilderRegistrationInvariants_RejectDuplicateAndLateOptionsRegistrations()
    {
        var services = TestServices.Create();
        VoiceAssistantBargeInOptions bargeIn = new();
        var duplicateBargeIn = VoiceAssistant.CreateBuilder().AddBargeIn(
            bargeIn,
            TestAcousticPathDelayEstimatorFactory.Instance);
        AssertCompositionCode(
            "VBX-COMP-002",
            () => duplicateBargeIn.AddBargeIn(
                bargeIn,
                TestAcousticPathDelayEstimatorFactory.Instance));

        var afterWakeup = VoiceAssistant.CreateBuilder().AddWakeup(services.Wakeup);
        AssertCompositionCode("VBX-COMP-002", () => afterWakeup.UseOptions(new VoiceAssistantOptions()));
        var afterSpeaker = VoiceAssistant.CreateBuilder().AddSpeakerVerification(services.Speaker);
        AssertCompositionCode("VBX-COMP-002", () => afterSpeaker.UseOptions(new VoiceAssistantOptions()));
        var afterBargeIn = VoiceAssistant.CreateBuilder().AddBargeIn(
            bargeIn,
            TestAcousticPathDelayEstimatorFactory.Instance);
        AssertCompositionCode("VBX-COMP-002", () => afterBargeIn.UseOptions(new VoiceAssistantOptions()));

        VoiceAssistantBuilder duplicateRecognition = VoiceAssistant.CreateBuilder()
            .UseSpeechRecognition(services.Asr);
        AssertCompositionCode("VBX-COMP-002", () => duplicateRecognition.UseSpeechRecognition(services.Asr));
    }

    [Fact]
    public void ModuleWiring_RejectsMissingEffectHandlersBeforeRuntimeStartup()
    {
        AssistantComposition composition = CreateRequiredDuplexComposition();
        AssistantCompositionException exception = Assert.Throws<AssistantCompositionException>(
            () => new ModuleWiring(composition.RuntimePlan, []));
        Assert.Equal("VBX-COMP-007", exception.Code);
    }

    private static void AssertZeroAudioFormatIsRejected(Action<FakeDuplexService, FakeDuplexService> configure)
    {
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        configure(record, playback);
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseAudio((IAudioRecordService)record, (IAudioPlayService)playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);

        AssertCompositionCode("VBX-COMP-005", () => builder.Build());
        Assert.Equal(0, record.StartCount);
        Assert.Equal(0, playback.StartCount);
    }

    private static void AssertBargeInCapabilityRejected(
        bool supportsSimultaneousCapturePlayback,
        bool supportsRenderReference,
        bool supportsProcessedCaptureTimeline)
    {
        var services = TestServices.Create();
        var audio = new CapabilityTestAudioFrontend(
            new FakeDuplexService(),
            supportsSimultaneousCapturePlayback,
            supportsRenderReference,
            supportsProcessedCaptureTimeline);
        var graph = new AssistantCapabilityGraph(
            AssistantCompositionProfile.BasicDuplexQa,
            audio,
            services.Vad,
            services.Asr,
            services.Llm,
            services.Tts,
            wakeup: null,
            speaker: null,
            acousticDiagnostics: null,
            new VoiceAssistantOptions { BargeIn = { Enable = true } });

        AssertCompositionCode("VBX-COMP-008", () => graph.ValidateAndCreateComposition());
        Assert.Equal(0, audio.Runtime.StartCount);
    }

    private static AssistantComposition CreateRequiredDuplexComposition()
    {
        var services = TestServices.Create();
        return VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();
    }

    private static void AssertCompositionCode(string code, Action action)
    {
        AssistantCompositionException exception =
            Assert.Throws<AssistantCompositionException>(action);
        Assert.Equal(code, exception.Code);
    }

    private sealed class CapabilityTestAudioFrontend(
        VoiceAssistantLifecycleTests.FakeDuplexService runtime,
        bool supportsSimultaneousCapturePlayback,
        bool supportsRenderReference,
        bool supportsProcessedCaptureTimeline) : IAssistantAudioFrontend
    {
        private readonly DuplexAssistantAudioFrontend _inner = new(runtime);

        public VoiceAssistantLifecycleTests.FakeDuplexService Runtime { get; } = runtime;

        public IAssistantAudioInput Input => _inner.Input;

        public IAssistantAudioOutput Output => _inner.Output;

        public IProcessedCaptureSource? ProcessedCapture =>
            SupportsProcessedCaptureTimeline ? _inner.ProcessedCapture : null;

        public IRenderReferenceSource? RenderReference =>
            SupportsRenderReference ? _inner.RenderReference : null;

        public IDuplexAudioFrameSource? DuplexFrames =>
            SupportsRenderReference ? _inner.DuplexFrames : null;

        public SpeakerCaptureDescriptor? SpeakerCapture =>
            SupportsRenderReference ? _inner.SpeakerCapture : null;

        public bool SupportsSimultaneousCapturePlayback { get; } = supportsSimultaneousCapturePlayback;

        public bool SupportsRenderReference { get; } = supportsRenderReference;

        public bool SupportsProcessedCaptureTimeline { get; } = supportsProcessedCaptureTimeline;

        public bool SupportsSpeakerCaptureTimeline => SupportsRenderReference;

        public Task StartAsync(CancellationToken cancellationToken = default)
            => _inner.StartAsync(cancellationToken);

        public Task StopAsync(CancellationToken cancellationToken = default)
            => _inner.StopAsync(cancellationToken);
    }

    private static void AssertBuilderMethod(Type builder, string name, params Type[] parameterTypes)
        => Assert.NotNull(builder.GetMethod(name, BindingFlags.Public | BindingFlags.Instance, parameterTypes));

    private static T Require<T>(T? value)
        where T : class
    {
        Assert.NotNull(value);
        return value!;
    }
}
