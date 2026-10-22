using System.Reflection;
using System.Text.Json;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Execution;
using VoiceBox.Assistant.Runtime.Effects.Handlers;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void CapabilityGraph_UsesClosedRequiresProvidesEdgesAndOwnershipSnapshot()
    {
        var services = TestServices.Create();
        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();

        Assert.Contains(composition.Snapshot.CapabilityEdges,
            edge => edge == "AudioFrontend:Provides:AudioCapture:0");
        Assert.Contains(composition.Snapshot.CapabilityEdges,
            edge => edge == "Recognition:Requires:TurnBoundary:0");
        Assert.Contains("StructuredToolUse", Enum.GetNames<AssistantCapability>());
        Assert.Contains(composition.Snapshot.Ownership,
            value => value.ModuleId == "AudioFrontend.Record" &&
                value.Ownership == ResourceOwnership.Borrowed);
        Assert.DoesNotContain(composition.RuntimePlan.Ownership,
            value => value.ModuleId == "AudioFrontend.Adapter");

        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        AssistantComposition split = VoiceAssistant.CreateBuilder()
            .UseAudio(record, playback)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();
        Assert.Contains(split.RuntimePlan.Ownership,
            value => value.ModuleId == "AudioFrontend.Adapter" &&
                value.Ownership == ResourceOwnership.AssistantOwned);
    }

    [Fact]
    public void RuntimeEffectRunner_BindsSevenDistinctCapabilityFamilyHandlers()
    {
        var services = TestServices.Create();
        using VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .Build();

        object runtime = typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        object effectRunner = RuntimeGraphProbe.RequireField(runtime, "_effects");
        var handlers = (AssistantEffectHandlers)effectRunner.GetType()
            .GetField("_handlers", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(effectRunner)!;

        object?[] handlerInstances =
        [
            handlers.InputGate,
            handlers.Recognition,
            handlers.ResponseGeneration,
            handlers.Playback,
            handlers.Conversation,
            handlers.Events,
            handlers.RuntimeEffects,
        ];

        Assert.DoesNotContain(handlerInstances, handler => handler is null);
        Assert.DoesNotContain(handlerInstances, handler => ReferenceEquals(handler, runtime));
        Assert.Equal(
            handlerInstances.Length,
            handlerInstances.Distinct(ReferenceEqualityComparer.Instance).Count());
    }

    [Fact]
    public void AudioFrontend_ExposesTypedOptionalDataPlanePortsWithoutRuntimeLookup()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IAssistantAudioInput"));
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IAssistantAudioOutput"));
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IRenderReferenceSource"));
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IProcessedCaptureSource"));
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IAcousticProcessingDiagnostics"));
        Assert.NotNull(assembly.GetType("VoiceBox.Assistant.Realtime.Routing.AudioFrameRouter"));
    }

    [Fact]
    public void AcousticProcessorAndMetricsSource_AreOrthogonalContracts()
    {
        Type processor = typeof(IAcousticEchoCancellationService);
        Type metricsSource = typeof(IAcousticProcessingMetricsSource);

        Assert.DoesNotContain(metricsSource, processor.GetInterfaces());
        Assert.DoesNotContain(
            processor.GetMethods(),
            method => method.Name.Contains("Metric", StringComparison.Ordinal));
        Assert.False(typeof(IDisposable).IsAssignableFrom(metricsSource));
        MethodInfo method = Assert.Single(metricsSource.GetMethods());
        Assert.Equal(nameof(IAcousticProcessingMetricsSource.TryGetLatestMetrics), method.Name);

        MethodInfo registration = typeof(VoiceAssistantBuilder).GetMethod(
            nameof(VoiceAssistantBuilder.AddAcousticDiagnostics))!;
        ParameterInfo parameter = Assert.Single(registration.GetParameters());
        Assert.Equal(metricsSource, parameter.ParameterType);
    }

    [Fact]
    public void Runtime_ConsumesTypedAudioAndTurnBoundaryPortsInsteadOfProviders()
    {
        string[] fieldTypes = ShellSurface.Fields
            .Select(static field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();

        // The shell must face the semantic ports, never a concrete audio or VAD provider.
        Assert.Contains("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IAssistantAudioFrontend", fieldTypes);
        Assert.Contains("VoiceBox.Assistant.Modules.AudioFrontend.Contracts.IAssistantAudioOutput", fieldTypes);
        Assert.Contains("VoiceBox.Assistant.Modules.TurnBoundary.ITurnBoundarySource", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Core.Services.AudioService.IAudioDuplexService", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Core.Services.AudioService.IAudioRecordService", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Core.Services.AudioService.IAudioPlayService", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Core.Services.VadService.IVoiceActivityDetectService", fieldTypes);

        // AEC metrics reach the runtime through the diagnostics port only.
        Assert.DoesNotContain(
            typeof(AecMetricsSampler).GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Select(static field => field.FieldType.FullName ?? field.FieldType.Name),
            value => value is
                "VoiceBox.Core.Services.AECService.IAcousticEchoCancellationService" or
                "VoiceBox.Core.Services.AcousticProcessing.IAcousticProcessingMetricsSource");
    }

    [Fact]
    public void Composition_OwnsRenderAwareDetectorSelectionAndOwnershipDrivenDisposal()
    {
        var services = TestServices.Create();
        AssistantComposition withoutBargeIn = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();
        Assert.Null(withoutBargeIn.RuntimePlan.PlaybackBargeInDetectorFactory);

        AssistantComposition withBargeIn = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddBargeIn(
                new VoiceAssistantBargeInOptions(),
                TestAcousticPathDelayEstimatorFactory.Instance)
            .BuildComposition();
        Assert.NotNull(withBargeIn.RuntimePlan.PlaybackBargeInDetectorFactory);
        Assert.Contains(
            withBargeIn.RuntimePlan.Ownership,
            value => value.ModuleId == "BargeIn.Module" &&
                value.Ownership == ResourceOwnership.AssistantOwned);

        // The lifecycle coordinator decides disposal from the plan, not from a type test.
        Assert.Contains(
            typeof(AssistantRuntimeLifecycleCoordinator).GetFields(
                BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == typeof(IReadOnlyList<ModuleOwnership>));
    }

    [Fact]
    public void NearEndAndDetector_SplitOwnershipIntoDedicatedStateTypes()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type coordinator = assembly.GetType("VoiceBox.Assistant.Modules.BargeIn.ActiveNearEndCoordinator")!;
        Type detector = assembly.GetType(
            "VoiceBox.Assistant.Modules.BargeIn.RenderAware.RenderAwarePlaybackBargeInDetector")!;

        // Candidate, capture session and endpoint policy each have one owner type.
        string[] coordinatorFieldTypes = coordinator
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(static field => field.FieldType.Name)
            .ToArray();
        Assert.Contains("NearEndEndpointPolicy", coordinatorFieldTypes);
        Assert.Contains("NearEndSpeakerCandidateState", coordinatorFieldTypes);
        Assert.Contains("NearEndCaptureSession", coordinatorFieldTypes);

        // The migrated state must no longer exist as flattened coordinator fields.
        string[] coordinatorFieldNames = coordinator
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(static field => field.Name)
            .ToArray();
        foreach (string retired in new[]
        {
            "_activeTurnId",
            "_directFeed",
            "_endDrainPending",
            "_speakerCandidateActive",
            "_endpointVadCandidateActive",
        })
        {
            Assert.DoesNotContain(retired, coordinatorFieldNames);
        }

        // The alignment stage is a dedicated pipeline collaborator.
        Assert.Contains(
            "RenderAlignmentStage",
            detector.GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Select(static field => field.FieldType.Name));
        Assert.DoesNotContain(
            "_lastRefinedEstimateSample",
            detector.GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Select(static field => field.Name));
    }

    [Fact]
    public void SpeakerVerifier_SplitsTimelineCacheSchedulerAndPolicyOwnership()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        // The streaming tracker and its queue/ledger are retired with the FeedAudio identity path.
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Modules.Speaker.ContinuousSpeakerTracker"));
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeakerAudioQueue"));
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeakerEvidenceLedger"));

        Type verifier = assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeechGatedSpeakerVerifier")!;
        string[] fieldTypes = verifier
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(static field => field.FieldType.Name)
            .ToArray();

        // Identity ring, evidence cache, bounded inference scheduler, speech-gated prefetch, window
        // geometry, quality gate and the validated policy are separate owners behind the verifier.
        Assert.Contains("SpeakerAudioTimelineBuffer", fieldTypes);
        Assert.Contains("SpeakerEvidenceCache", fieldTypes);
        Assert.Contains("SpeakerInferenceScheduler", fieldTypes);
        Assert.Contains("SpeakerSpeechPrefetchGate", fieldTypes);
        Assert.Contains("SpeakerEvidenceWindowSelector", fieldTypes);
        Assert.Contains("SpeakerWindowQualityEvaluator", fieldTypes);
        Assert.Contains("SpeakerVerificationPolicy", fieldTypes);

        // The verifier holds owner references only: no raw evidence map, no worker task, no channel,
        // no duplicated window geometry.
        string[] fieldNames = verifier
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(static field => field.Name)
            .ToArray();
        Assert.DoesNotContain("_evidence", fieldNames);
        Assert.DoesNotContain("_worker", fieldNames);
        Assert.DoesNotContain("_windowSamples", fieldNames);
        Assert.DoesNotContain("_hopSamples", fieldNames);
        Assert.DoesNotContain(fieldTypes, value => value.Contains("Channel", StringComparison.Ordinal));
        Assert.DoesNotContain(fieldTypes, value => value.Contains("Dictionary", StringComparison.Ordinal) && value != "IReadOnlyDictionary`2");

        // The scheduler is the only owner of the provider worker; the prefetch gate never calls the
        // provider itself.
        Type scheduler = assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeakerInferenceScheduler")!;
        Assert.Contains("_worker", scheduler.GetFields(BindingFlags.Instance | BindingFlags.NonPublic).Select(static field => field.Name));
        Type prefetch = assembly.GetType("VoiceBox.Assistant.Modules.Speaker.SpeakerSpeechPrefetchGate")!;
        Assert.DoesNotContain(
            prefetch.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            static field => field.FieldType.Name == "ISpeakerService");
    }

    [Fact]
    public void WakeupGate_SplitsModePreRollAuthorizationAndServiceAdapter()
    {
        Type gate = typeof(VoiceAssistant).Assembly.GetType(
            "VoiceBox.Assistant.Modules.Wakeup.WakeupSessionGate")!;
        string[] fieldTypes = gate
            .GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
            .Select(static field => field.FieldType.Name)
            .ToArray();

        Assert.Contains("WakeupModeState", fieldTypes);
        Assert.Contains("WakeupPreRollTimeline", fieldTypes);
        Assert.Contains("WakeupSpeakerAuthorization", fieldTypes);
        Assert.Contains("WakeupDetectorAdapter", fieldTypes);
    }

    [Fact]
    public void RuntimeTelemetry_UsesStructuredEventIdsAndKeepsTracingOutOfRuntimeAndKernel()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type eventIds = assembly.GetType("VoiceBox.Assistant.Runtime.AssistantRuntimeEventIds")
            ?? throw new Xunit.Sdk.XunitException("AssistantRuntimeEventIds is missing.");

        Assert.Equal(
            ["Started", "Stopped", "TurnCompleted", "TerminalFault"],
            eventIds.GetFields(BindingFlags.Public | BindingFlags.Static)
                .Select(static field => field.Name)
                .ToArray());

        // Hosting is co-located with the core assembly. Its diagnostics references are confined
        // to the Hosting source boundary rather than a separate assembly.
        Assert.Contains(
            assembly.GetReferencedAssemblies(),
            reference => reference.Name == "System.Diagnostics.DiagnosticSource");

        // Every runtime log call that carries an EventId must use the structured catalog.
        // The lifecycle owner logs start/stop; the turn owner logs completion.
        string lifecycle = ReadRuntimeSource("Lifecycle", "Controllers", "AssistantRuntimeLifecycle.cs");
        Assert.Contains("AssistantRuntimeEventIds.Started", lifecycle, StringComparison.Ordinal);
        Assert.Contains("AssistantRuntimeEventIds.Stopped", lifecycle, StringComparison.Ordinal);
        string turnCompletion = ReadRuntimeSource("TurnContext", "TurnCompletionCoordinator.cs");
        Assert.Contains("AssistantRuntimeEventIds.TurnCompleted", turnCompletion, StringComparison.Ordinal);
        string faults = ReadRuntimeSource("Lifecycle", "Controllers", "AssistantFaultConverger.cs");
        Assert.Contains("AssistantRuntimeEventIds.TerminalFault", faults, StringComparison.Ordinal);
        string runtime = lifecycle + turnCompletion + faults;
        Assert.Contains("AssistantRuntimeEventIds.TerminalFault", runtime, StringComparison.Ordinal);

        string hosting = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "src",
            "VoiceBox.Assistant",
            "Hosting",
            "Lifecycle",
            "VoiceAssistantHostedLifecycle.cs"));
        Assert.Contains("new VoiceAssistantRuntimeTelemetryObserver(_assistant)", hosting, StringComparison.Ordinal);
        string telemetry = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "src",
            "VoiceBox.Assistant",
            "Hosting",
            "Telemetry",
            "VoiceAssistantHostingTelemetry.cs"));
        Assert.Contains("ActivitySource ActivitySource = new(\"VoiceBox.Assistant.Hosting\")", telemetry, StringComparison.Ordinal);
        Assert.Contains("Meter Meter = new(\"VoiceBox.Assistant.Hosting\")", telemetry, StringComparison.Ordinal);
    }

    [Fact]
    public void StructuredToolUse_IsAValidatedCapabilityThatNoProfileProvides()
    {
        var services = TestServices.Create();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);
        builder.RequireCapability(AssistantCapability.StructuredToolUse);

        AssistantCompositionException failure =
            Assert.Throws<AssistantCompositionException>(builder.Build);
        Assert.Equal("VBX-COMP-011", failure.Code);
        Assert.Equal(0, services.Audio.StartCount);
    }

    [Fact]
    public void BuilderWithoutBargeIn_DoesNotConstructActiveNearEndCoordinator()
    {
        var services = TestServices.Create();
        using VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .Build();

        object runtime = typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        object? activeNearEnd = RuntimeGraphProbe.FindField(runtime, "_activeNearEnd");

        Assert.Null(activeNearEnd);
        Assert.Null(RuntimeGraphProbe.FindField(runtime, "_validationRunner"));
        Assert.DoesNotContain(
            runtime.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Select(static field => field.FieldType.FullName ?? field.FieldType.Name),
            value => value is
                "VoiceBox.Core.Services.ASRService.IASRService" or
                "VoiceBox.Core.Services.LLMService.IStreamLLMService" or
                "VoiceBox.Core.Services.WakeupService.IWakeupService" or
                "VoiceBox.Core.Services.SpeakerService.ISpeakerService");
    }

    [Fact]
    public void OperationCorrelationGate_RejectsDuplicateAndCrossTurnCompletion()
    {
        var gate = new OperationCorrelationGate();

        Assert.True(gate.TryAdmit(1, 10, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(1, 10, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(2, 10, AssistantSignalKind.RecognitionCompleted));

        gate.ForgetTurn(1);
        Assert.False(gate.TryAdmit(1, 11, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(2, 10, AssistantSignalKind.RecognitionCompleted));
        Assert.True(gate.TryAdmit(2, 11, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(0, 11, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(3, 0, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(-1, 12, AssistantSignalKind.RecognitionCompleted));
        Assert.False(gate.TryAdmit(3, -1, AssistantSignalKind.RecognitionCompleted));
        Assert.True(gate.TryAdmit(0, 13, AssistantSignalKind.WakeupModeChanged));
    }

    [Fact]
    public void TurnObservationOwners_KeepFeatureResourcesWithTheirSoleModuleOwner()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type recognitionRouter = assembly.GetType(
            "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionTurnOwner")!;
        Type recognitionOwner = assembly.GetType(
            "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionModule")!;
        Type validationOwner = assembly.GetType(
            "VoiceBox.Assistant.Modules.BargeIn.BargeInValidationTurnOwner")!;

        Assert.NotNull(recognitionRouter);
        Assert.NotNull(recognitionOwner);
        Assert.NotNull(validationOwner);

        FieldInfo[] routerFields = recognitionRouter.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.DoesNotContain(routerFields, field => field.FieldType == typeof(ITurnResourcePort));
        Assert.DoesNotContain(routerFields, field => field.FieldType == typeof(CancellationTokenSource));
        Assert.DoesNotContain(routerFields, field =>
            field.FieldType == typeof(List<float[]>) ||
            field.FieldType == typeof(Queue<float[]>));
        Assert.Contains(routerFields, field => field.FieldType.Name == "IRecognitionCaptureOwner");

        FieldInfo[] recognitionFields = recognitionOwner.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.Contains(recognitionFields, field => field.Name == "_captureByTurn");
        Assert.Contains(recognitionFields, field => field.Name == "_sessionsByTurn");

        Type capturePort = assembly.GetType(
            "VoiceBox.Assistant.Modules.Recognition.Contracts.IRecognitionCaptureOwner")!;
        Assert.DoesNotContain(
            capturePort.GetMethods().SelectMany(static method => method.GetParameters()),
            parameter => typeof(Delegate).IsAssignableFrom(parameter.ParameterType));
        Assert.DoesNotContain(
            capturePort.GetMethods().SelectMany(static method => method.GetParameters()),
            parameter => parameter.ParameterType.Name == "ICandidatePreRollSource");

        FieldInfo[] validationFields = validationOwner.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.Contains(validationFields, field => field.Name == "_responseCts");
        Assert.Contains(validationFields, field => field.Name == "_pendingAudio");
        Assert.DoesNotContain(validationFields, field => field.FieldType == typeof(ITurnResourcePort));
    }

    /// <summary>
    /// Pairwise coverage of the optional capability modules over both audio transports.
    /// Confirmation defaults to off, so KWS without a speaker provider is legal and an installed
    /// speaker provider without Confirmation.Enable installs no online identity module. Barge-in
    /// over split audio must fail because split endpoints cannot declare a render reference, and
    /// KWS over split audio fails because there is no processed capture timeline (VBX-COMP-009).
    /// </summary>
    [Theory]
    // duplex: all pairs of (wakeup, speaker, bargeIn) plus the full stack.
    [InlineData(false, false, false, false, null)]
    [InlineData(false, true, false, false, null)]
    [InlineData(false, false, true, false, null)]
    [InlineData(false, false, false, true, null)]
    [InlineData(false, true, true, false, null)]
    [InlineData(false, true, false, true, null)]
    [InlineData(false, false, true, true, null)]
    [InlineData(false, true, true, true, null)]
    // split audio: valid without barge-in or KWS, rejected with either.
    [InlineData(true, false, false, false, null)]
    [InlineData(true, false, true, false, null)]
    [InlineData(true, false, false, true, "VBX-COMP-008")]
    [InlineData(true, true, false, false, "VBX-COMP-009")]
    [InlineData(true, true, true, false, "VBX-COMP-009")]
    public void OptionalCapabilityPairs_BuildOrFailDeterministically(
        bool splitAudio,
        bool wakeup,
        bool speaker,
        bool bargeIn,
        string? expectedErrorCode)
    {
        var services = TestServices.Create();
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();

        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder();
        if (splitAudio)
        {
            builder.UseAudio(
                (VoiceBox.Core.Services.AudioService.IAudioRecordService)record,
                (VoiceBox.Core.Services.AudioService.IAudioPlayService)playback);
        }
        else
        {
            builder.UseAudio(services.Audio);
        }

        builder
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);

        if (wakeup)
            builder.AddWakeup(services.Wakeup);
        if (speaker)
            builder.AddSpeakerVerification(services.Speaker);
        if (bargeIn)
        {
            builder.AddBargeIn(
                new VoiceAssistantBargeInOptions(),
                TestAcousticPathDelayEstimatorFactory.Instance);
        }

        if (expectedErrorCode is not null)
        {
            AssistantCompositionException failure =
                Assert.Throws<AssistantCompositionException>(builder.BuildComposition);
            Assert.Equal(expectedErrorCode, failure.Code);
            Assert.Equal(0, services.Audio.StartCount);
            Assert.Equal(0, record.StartCount);
            Assert.Equal(0, playback.StartCount);
            return;
        }

        AssistantComposition composition = builder.BuildComposition();
        Assert.Equal(
            splitAudio ? "BasicSplitAudioQa" : "BasicDuplexQa",
            composition.Snapshot.ProfileId);

        // A module appears in the lifecycle plan and the capability snapshot only when registered.
        // Online speaker verification is off unless Confirmation.Enable is set; this matrix leaves
        // it at the default (false), so a registered provider installs nothing.
        bool identityEffective = false;
        Assert.Equal(wakeup, composition.Lifecycle.StartOrder.Contains(AssistantLifecycleParticipant.WakeupGate));
        Assert.Equal(identityEffective, composition.Lifecycle.StartOrder.Contains(AssistantLifecycleParticipant.SpeakerEvidence));
        Assert.Equal(bargeIn, composition.Lifecycle.StartOrder.Contains(AssistantLifecycleParticipant.BargeIn));
        Assert.Equal(wakeup, composition.Snapshot.Capabilities.Contains("WakeupGate"));
        Assert.Equal(identityEffective, composition.Snapshot.Capabilities.Contains("SpeakerEvidence"));
        Assert.Equal(bargeIn, composition.Snapshot.Capabilities.Contains("BargeIn"));
        Assert.Equal(identityEffective, composition.RuntimePlan.Speaker is not null);
        if (identityEffective)
        {
            // The speaker module is started before the audio frontend so no identity frame is lost.
            List<AssistantLifecycleParticipant> order = composition.Lifecycle.StartOrder.ToList();
            Assert.True(
                order.IndexOf(AssistantLifecycleParticipant.SpeakerEvidence) <
                order.IndexOf(AssistantLifecycleParticipant.AudioFrontend));
        }

        // Building must never start an endpoint.
        Assert.Equal(0, services.Audio.StartCount);
        Assert.Equal(0, record.StartCount);
        Assert.Equal(0, playback.StartCount);
    }

    [Fact]
    public void PublicEventEffects_AreBoundToDedicatedTypedHandler()
    {
        Assert.Contains(ShellSurface.Fields, field =>
            field.FieldType == typeof(AssistantRuntimePublicEventEffectHandler));

        Assert.DoesNotContain(
            typeof(AssistantRuntime).GetInterfaces(),
            type => type == typeof(IPublicEventEffectHandler));

        string source = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "src",
            "VoiceBox.Assistant",
            "Runtime",
            "Effects",
            "Execution",
            "AssistantEffectHandlers.cs"));
        Assert.Contains("IPublicEventEffectHandler events", source, StringComparison.Ordinal);
        string factory = File.ReadAllText(Path.Combine(
            FindRepositoryRoot(),
            "src",
            "VoiceBox.Assistant",
            "Composition",
            "Wiring",
            "AssistantRuntimeFactory.cs"));
        Assert.Contains("plan.Speaker.Create", factory, StringComparison.Ordinal);
        Assert.Contains("plan.Wakeup.Create", factory, StringComparison.Ordinal);
        Assert.Contains("public IPublicEventEffectHandler Events { get; }", source, StringComparison.Ordinal);
        // The aggregate handler interface must stay deleted so no single object can claim every
        // capability family again.
        Assert.Null(typeof(VoiceAssistant).Assembly.GetType(
            "VoiceBox.Assistant.Composition.IAssistantEffectHandlerSet"));
    }

    [Fact]
    public void KernelSnapshot_HasOneRuntimeCommitSiteAndStateMachineDoesNotInvokeReducer()
    {
        string root = FindRepositoryRoot();
        string runtime = ReadRuntimeSource("AssistantRuntime.cs");

        // The facade neither holds the snapshot nor reduces it, and constructs no capability.
        Assert.DoesNotContain("_kernelSnapshot", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("KernelCommitResult", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("new SpeakerModule", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("new WakeupModule", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("new LlmResponseGenerator", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("_lifecycleState", runtime, StringComparison.Ordinal);

        // Exactly one commit site publishes the snapshot, and it is the control-loop writer.
        string writer = ReadRuntimeSource("Control", "Controllers", "AssistantControlLoopTurnWriter.cs");
        Assert.Contains("_snapshots.Publish(decision.Next)", writer, StringComparison.Ordinal);
        Assert.Contains("TurnKernel.Reduce", writer, StringComparison.Ordinal);

        string snapshotStore = ReadRuntimeSource("Control", "State", "TurnSnapshotStore.cs");
        Assert.DoesNotContain("TurnKernel", snapshotStore, StringComparison.Ordinal);

        // The turn-observation owners classify facts; they never reduce or hold turn ownership.
        foreach (string owner in new[]
        {
            Path.Combine(root, "src", "VoiceBox.Assistant", "Modules", "BargeIn", "Controllers", "SpeechCandidateOwner.cs"),
            Path.Combine(root, "src", "VoiceBox.Assistant", "Modules", "BargeIn", "Controllers", "BargeInValidationTurnOwner.cs"),
            Path.Combine(root, "src", "VoiceBox.Assistant", "Modules", "Recognition", "Controllers", "RecognitionTurnOwner.cs"),
        })
        {
            string source = File.ReadAllText(owner);
            Assert.DoesNotContain("TurnKernel", source, StringComparison.Ordinal);
            Assert.DoesNotContain("KernelCommitResult", source, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void ResponseModule_DoesNotFormatUserTextIntoLogs()
    {
        string root = FindRepositoryRoot();
        string source = File.ReadAllText(Path.Combine(
            root,
            "src",
            "VoiceBox.Assistant",
            "Modules",
            "Response",
            "Controllers",
            "ResponseModule.cs"));

        Assert.DoesNotContain("('{Text}')", source, StringComparison.Ordinal);
        Assert.DoesNotContain("rawText);", source, StringComparison.Ordinal);
        Assert.DoesNotContain("userText);", source, StringComparison.Ordinal);
        Assert.DoesNotContain("BargeIn.Enable", source, StringComparison.Ordinal);
    }

}
