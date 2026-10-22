using System.Reflection;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Runtime.Contracts;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Guards the complete Runtime/Modules/Composition closure against rebuilding the retired host
/// callback service network. The allowlist is member-specific and contains only provider events,
/// closed algorithm/work factories, and the public speaker-enrollment interaction callback.
/// </summary>
public sealed class BroadCallbackDependencyClosureTests
{
    private const BindingFlags Declared = BindingFlags.Instance | BindingFlags.Static |
        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;

    private static readonly HashSet<string> AllowedDelegateFields =
    [
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner._processedCapture",
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner._duplexFrame",
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner._speechStarted",
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner._speaking",
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner._speechEnded",
        "VoiceBox.Assistant.Modules.TurnBoundary.VadTurnBoundaryAdapter._speaking",
        "VoiceBox.Assistant.Modules.TurnBoundary.VadTurnBoundaryAdapter._speechEnded",
        "VoiceBox.Assistant.Modules.BargeIn.RenderAware.RenderAwarePlaybackBargeInDetector._resampledPairHandler",
    ];

    private static readonly HashSet<string> AllowedDelegateMembers =
    [
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner..ctor",
        "VoiceBox.Assistant.Runtime.Effects.IEffectBackgroundScheduler.Schedule",
        "VoiceBox.Assistant.Runtime.Lifecycle.AssistantTaskSupervisor.VoiceBox.Assistant.Runtime.Effects.IEffectBackgroundScheduler.Schedule",
        "VoiceBox.Assistant.Modules.Speaker.SpeakerEnrollmentCoordinator.EnrollAsync",
        "VoiceBox.Assistant.Modules.Speaker.SpeakerEnrollmentCoordinator.EnrollCoreAsync",
        "VoiceBox.Assistant.Modules.Wakeup.WakeupDetectorAdapter.Subscribe",
        "VoiceBox.Assistant.Modules.Wakeup.WakeupDetectorAdapter.Unsubscribe",
    ];

    [Fact]
    public void ProductionClosure_HasNoUnapprovedStoredOrInjectedDelegates()
    {
        Type[] types = typeof(VoiceAssistant).Assembly.GetTypes()
            .Where(IsScannedProductionType)
            .Where(static type => type.GetCustomAttribute<CompilerGeneratedAttribute>() is null)
            .Where(static type => !typeof(IAsyncStateMachine).IsAssignableFrom(type))
            .ToArray();
        var violations = new List<string>();

        foreach (Type type in types)
        {
            foreach (FieldInfo field in type.GetFields(Declared))
            {
                if (!typeof(Delegate).IsAssignableFrom(field.FieldType) ||
                    IsEventBackingField(type, field) ||
                    AllowedDelegateFields.Contains(MemberKey(type, field.Name)))
                {
                    continue;
                }

                violations.Add($"field:{MemberKey(type, field.Name)}:{field.FieldType}");
            }

            foreach (ConstructorInfo constructor in type.GetConstructors(Declared))
                CheckCallable(type, constructor, ".ctor", violations);

            foreach (MethodInfo method in type.GetMethods(Declared).Where(static method => !method.IsSpecialName))
            {
                CheckCallable(type, method, method.Name, violations);
                if (typeof(Delegate).IsAssignableFrom(method.ReturnType) &&
                    !AllowedDelegateMembers.Contains(MemberKey(type, method.Name)))
                {
                    violations.Add($"return:{MemberKey(type, method.Name)}:{method.ReturnType}");
                }
            }

            foreach (PropertyInfo property in type.GetProperties(Declared))
            {
                if (typeof(Delegate).IsAssignableFrom(property.PropertyType))
                    violations.Add($"property:{MemberKey(type, property.Name)}:{property.PropertyType}");
            }
        }

        Assert.True(violations.Count == 0, string.Join(Environment.NewLine, violations));
    }

    [Fact]
    public void NearEndAndRealtimePorts_DoNotExposeCancellationSourcesOrLateBinding()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Modules.BargeIn.IActiveNearEndModule"));
        Assert.DoesNotContain(
            assembly.GetTypes().SelectMany(static type => type.GetMethods(Declared)),
            static method => method.Name == "BindActiveNearEnd");

        Type[] realtimePorts =
        [
            typeof(INearEndCapturePort),
            typeof(INearEndRenderPort),
            typeof(INearEndEndpointPort),
            typeof(INearEndConfirmationStore),
            typeof(IRuntimeLivenessView),
            typeof(IPlaybackActivityView),
        ];
        Assert.All(realtimePorts, port => Assert.DoesNotContain(
            port.GetMethods(),
            static method => method.ReturnType == typeof(CancellationTokenSource) ||
                method.GetParameters().Any(parameter =>
                    parameter.ParameterType == typeof(CancellationTokenSource) ||
                    parameter.ParameterType == typeof(CancellationTokenSource).MakeByRefType())));
    }

    private static void CheckCallable(
        Type type,
        MethodBase callable,
        string name,
        List<string> violations)
    {
        string key = MemberKey(type, name);
        if (AllowedDelegateMembers.Contains(key))
            return;

        foreach (ParameterInfo parameter in callable.GetParameters())
        {
            if (typeof(Delegate).IsAssignableFrom(parameter.ParameterType))
                violations.Add($"parameter:{key}:{parameter.Name}:{parameter.ParameterType}");
        }
    }

    private static bool IsScannedProductionType(Type type)
        => type.Namespace?.StartsWith("VoiceBox.Assistant.Runtime", StringComparison.Ordinal) == true ||
            type.Namespace?.StartsWith("VoiceBox.Assistant.Modules", StringComparison.Ordinal) == true ||
            type.Namespace?.StartsWith("VoiceBox.Assistant.Composition", StringComparison.Ordinal) == true;

    private static bool IsEventBackingField(Type type, FieldInfo field)
        => type.GetEvents(Declared).Any(value =>
            value.Name == field.Name && value.EventHandlerType == field.FieldType);

    private static string MemberKey(Type type, string name) => $"{type.FullName}.{name}";
}
