using System.Reflection;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class LifecycleShellContractTests
{
    [Fact]
    public void LifecycleShell_ExposesClosedLifecycleStateSubscriptionCoordinatorAndSupervisorContracts()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;

        Type state = Require(assembly, "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeLifecycleState");
        Assert.True(state.IsEnum);
        Assert.Equal(
            ["Created", "Starting", "Running", "Stopping", "Stopped", "Faulted", "Disposed"],
            Enum.GetNames(state));

        Type stateMachine = Require(assembly, "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeLifecycleStateMachine");
        Assert.NotNull(stateMachine.GetProperty("State", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(stateMachine.GetMethod("BeginStart", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(stateMachine.GetMethod("CompleteStart", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(stateMachine.GetMethod("BeginStop", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(stateMachine.GetMethod("CompleteStop", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(stateMachine.GetMethod("MarkDisposed", BindingFlags.Instance | BindingFlags.Public));

        Type subscriptions = Require(assembly, "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeSubscriptionOwner");
        Assert.NotNull(subscriptions.GetProperty("IsAttached", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(subscriptions.GetMethod("Attach", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(subscriptions.GetMethod("Detach", BindingFlags.Instance | BindingFlags.Public));

        Type supervisor = Require(assembly, "VoiceBox.Assistant.Runtime.Lifecycle.AssistantTaskSupervisor");
        Assert.NotNull(supervisor.GetProperty("ActiveTaskCount", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(supervisor.GetMethod("Track", BindingFlags.Instance | BindingFlags.Public));
        Assert.NotNull(supervisor.GetMethod("WaitForCompletionAsync", BindingFlags.Instance | BindingFlags.Public));

        Type coordinator = Require(assembly, "VoiceBox.Assistant.Runtime.Lifecycle.AssistantRuntimeLifecycleCoordinator");
        Assert.DoesNotContain(coordinator.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public),
            static field => field.FieldType.Name is "AssistantCapabilityGraph" or "IServiceProvider" or "IAudioServiceManager");
        Assert.Contains(coordinator.GetConstructors(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic),
            static constructor => constructor.GetParameters().Any(
                static parameter => parameter.ParameterType.Name == "ModuleLifecyclePlan"));
    }

    [Fact]
    public void LifecycleStateMachine_StopDuringStart_CannotCompleteRunningAndConvergesStopped()
    {
        var lifecycle = new AssistantRuntimeLifecycleStateMachine();

        Assert.True(lifecycle.BeginStart());
        Assert.Equal(AssistantRuntimeLifecycleState.Starting, lifecycle.State);
        Assert.True(lifecycle.BeginStop());
        Assert.False(lifecycle.CompleteStart());

        lifecycle.FailStart();
        Assert.Equal(AssistantRuntimeLifecycleState.Stopping, lifecycle.State);

        lifecycle.CompleteStop();
        Assert.Equal(AssistantRuntimeLifecycleState.Stopped, lifecycle.State);
        Assert.False(lifecycle.BeginStart());
        Assert.Equal(AssistantRuntimeLifecycleState.Stopped, lifecycle.State);
    }

    private static Type Require(Assembly assembly, string name)
        => assembly.GetType(name)
            ?? throw new Xunit.Sdk.XunitException($"Required lifecycle contract type '{name}' was not found.");
}
