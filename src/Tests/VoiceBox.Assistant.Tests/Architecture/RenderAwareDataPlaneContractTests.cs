using System.Reflection;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Adapters;
using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Realtime.Timeline;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RenderAwareDataPlaneContractTests
{
    [Fact]
    public void RuntimeUsesConsumerSpecificNearEndPorts_AndDataPlaneOwnsNoKernelState()
    {
        FieldInfo[] runtimeFields = ShellSurface.Fields;

        Assert.Contains(runtimeFields, field => field.FieldType == typeof(INearEndCapturePort));
        Assert.Contains(runtimeFields, field => field.FieldType == typeof(INearEndRenderPort));
        Assert.Contains(runtimeFields, field => field.FieldType == typeof(INearEndEndpointPort));
        Assert.DoesNotContain(runtimeFields, field => field.FieldType.Name == "IActiveNearEndModule");
        Assert.DoesNotContain(runtimeFields, field => field.FieldType == typeof(NearEndDataPlane));

        Type[] dataPlaneTypes =
        [
            typeof(NearEndDataPlane),
            typeof(NearEndEffectExecutor),
            typeof(NearEndTimelineBuffer),
            typeof(NearEndEndpointTracker),
            typeof(NearEndCandidateStateMachine),
            typeof(NearEndCaptureStateMachine),
            typeof(NearEndConfirmationJournal),
            typeof(NearEndReplayBuffer),
            typeof(PlaybackDuckingPolicy),
            typeof(RenderAwarePlaybackBargeInDetector),
            typeof(RenderAwareDtdEvidencePolicy),
            typeof(BargeInDecisionAccumulator),
        ];
        foreach (Type type in dataPlaneTypes)
        {
            Assert.DoesNotContain(
                type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public),
                field => field.FieldType == typeof(AssistantRuntime) ||
                    field.FieldType == typeof(TurnSnapshot) ||
                    field.FieldType == typeof(IAudioDuplexService) ||
                    field.FieldType.Name.Contains("IServiceProvider", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void NearEndFacetsAreTypedDataPlaneBoundaries()
    {
        Type[] facets =
        [
            typeof(INearEndCapturePort),
            typeof(INearEndRenderPort),
            typeof(INearEndEndpointPort),
            typeof(INearEndConfirmationStore),
            typeof(ISpeechTimelineLocator),
            typeof(INearEndCaptureControlPort),
            typeof(INearEndResetPort),
        ];
        Assert.All(facets, facet => Assert.True(facet.IsInterface));
        PropertyInfo[] ports = typeof(NearEndPorts).GetProperties();
        Assert.All(facets, facet => Assert.Contains(ports, port => port.PropertyType == facet));

        MethodInfo duplex = typeof(INearEndRenderPort).GetMethod(nameof(INearEndRenderPort.ObserveDuplexAudio))!;
        Assert.Equal(typeof(DuplexAudioFrame).MakeByRefType(), duplex.GetParameters()[0].ParameterType);
        Assert.Equal(typeof(ProcessedCaptureFrame).MakeByRefType(),
            typeof(INearEndCapturePort).GetMethod(nameof(INearEndCapturePort.ObserveRecognitionAudio))!
                .GetParameters()[0].ParameterType);
        Assert.All(facets, facet => Assert.DoesNotContain(
            facet.GetMethods(),
            method => method.GetParameters().Any(parameter =>
                parameter.ParameterType == typeof(IAudioDuplexService) ||
                parameter.ParameterType == typeof(AssistantRuntime))));
    }

    [Fact]
    public void NearEndDataPlane_UsesOnlySmallTypedCollaboratorsAndOneEffectExecutor()
    {
        FieldInfo[] fields = typeof(NearEndDataPlane).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.DoesNotContain(fields, field => typeof(Delegate).IsAssignableFrom(field.FieldType));

        ConstructorInfo constructor = Assert.Single(typeof(NearEndDataPlane).GetConstructors());
        Assert.DoesNotContain(
            constructor.GetParameters(),
            parameter => typeof(Delegate).IsAssignableFrom(parameter.ParameterType));

        Type[] requiredCollaborators =
        [
            typeof(IActiveNearEndConversationView),
            typeof(IActiveNearEndDecisionSource),
            typeof(IRuntimeLivenessView),
            typeof(INearEndAuthorizationPolicy),
            typeof(NearEndEffectExecutor),
        ];
        Assert.All(requiredCollaborators, port => Assert.Contains(fields, field => field.FieldType == port));

        // Playback, control ingress and recognition are reachable only through the executor.
        Assert.DoesNotContain(fields, field => field.FieldType == typeof(IActiveNearEndPlaybackPort));
        Assert.DoesNotContain(fields, field => field.FieldType == typeof(IActiveNearEndSignalSink));
        Assert.DoesNotContain(fields, field => field.FieldType == typeof(INearEndRecognitionAudioSink));

        FieldInfo[] executorFields = typeof(NearEndEffectExecutor).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.DoesNotContain(executorFields, field => typeof(Delegate).IsAssignableFrom(field.FieldType));
        Assert.Contains(executorFields, field => field.FieldType == typeof(IActiveNearEndPlaybackPort));
        Assert.Contains(executorFields, field => field.FieldType == typeof(IActiveNearEndSignalSink));
        Assert.Contains(executorFields, field => field.FieldType == typeof(INearEndRecognitionAudioSink));
    }

    [Fact]
    public void RenderAwarePipelineUsesImmutableSnapshotsAndExtractedTimelineResampler()
    {
        Assert.True(typeof(TimelinePairResampler).IsClass);
        Assert.DoesNotContain(
            typeof(RenderAwarePlaybackBargeInDetector).GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic),
            nested => nested.Name.Contains("Resampler", StringComparison.Ordinal));

        Type[] immutableSnapshots =
        [
            typeof(RenderAwareBargeInSnapshot),
            typeof(ActiveNearEndDiagnosticSnapshot),
            typeof(MainAudioObservation),
            typeof(ActiveNearEndProcessingState),
        ];
        foreach (Type type in immutableSnapshots)
        {
            Assert.True(type.IsValueType, type.FullName);
            Assert.All(
                type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public),
                field => Assert.True(field.IsInitOnly, $"{type.Name}.{field.Name} must be immutable."));
        }
    }

    [Fact]
    public void TimelineAndBufferComponentsAreBoundedAndStreamLocal()
    {
        Type[] boundedComponents =
        [
            typeof(BoundedFractionalRenderBuffer),
            typeof(NearEndAudioReservoir),
            typeof(NearEndTimelineBuffer),
            typeof(NearEndReplayBuffer),
            typeof(TimelineMonoResampler),
            typeof(TimelinePairResampler),
        ];

        foreach (Type type in boundedComponents)
        {
            Assert.DoesNotContain(
                type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
                field => field.FieldType == typeof(IAudioDuplexService) ||
                    field.FieldType == typeof(AssistantRuntime) ||
                    field.FieldType.Name.Contains("Dictionary", StringComparison.Ordinal));
        }
    }
}
