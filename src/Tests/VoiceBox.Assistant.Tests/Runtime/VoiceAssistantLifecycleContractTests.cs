using Microsoft.Extensions.Logging;
using System.Reflection;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Modules.TurnBoundary.Adapters;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void SubscriptionOwner_AttachAndDetachAreExactlyOnce()
    {
        var services = TestServices.Create();
        int speechStarted = 0;
        var frontend = new DuplexAssistantAudioFrontend(services.Audio);
        var owner = new AssistantRuntimeSubscriptionOwner(
            frontend.ProcessedCapture,
            frontend.RenderReference,
            new VadTurnBoundaryAdapter(services.Vad),
            IgnoreProcessedCapture,
            IgnoreDuplexAudio,
            (_, _) => speechStarted++,
            (_, _) => { },
            (_, _) => { });

        owner.Attach();
        owner.Attach();
        services.Vad.EmitSpeechStarted();

        Assert.True(owner.IsAttached);
        Assert.Equal(1, speechStarted);

        owner.Detach();
        owner.Detach();
        services.Vad.EmitSpeechStarted();

        Assert.False(owner.IsAttached);
        Assert.Equal(1, speechStarted);
    }

    [Fact]
    public async Task Stop_LeavesNoShellSubscriptionRegisteredTaskOrRunCts()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        await assistant.StartAsync();
        await assistant.StopAsync();

        object runtime = typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        object subscriptions = RuntimeGraphProbe.RequireField(runtime, "_subscriptions");
        object supervisor = RuntimeGraphProbe.RequireField(runtime, "_tasks");

        Assert.False((bool)subscriptions.GetType().GetProperty("IsAttached")!.GetValue(subscriptions)!);
        Assert.Equal(0, (int)supervisor.GetType().GetProperty("ActiveTaskCount")!.GetValue(supervisor)!);
        object runScope = RuntimeGraphProbe.RequireField(runtime, "_runScope");
        Assert.Null(runScope.GetType()
            .GetField("_runCts", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(runScope));
    }

    [Fact]
    public async Task LifecycleTrace_StartStopDispose_IsExactAndSeparateFromTurnWork()
    {
        var services = TestServices.Create();
        VoiceAssistant assistant = services.CreateAssistant();
        var trace = new List<string>();
        assistant.StateChanged += (_, value) => trace.Add($"state:{value.From}->{value.To}");

        await assistant.StartAsync();
        trace.Add("start-return");
        await assistant.StopAsync();
        trace.Add("stop-return");
        assistant.Dispose();
        trace.Add("dispose-return");

        Assert.Equal(
            [
                "start-return",
                "state:Idle->Stopped",
                "stop-return",
                "dispose-return",
            ],
            trace);
        Assert.Equal(1, services.Audio.StartCount);
        Assert.Equal(1, services.Audio.StopCount);
    }

    [Fact]
    public void LifecycleCoordinator_DisposesAssistantOwnedBargeInExactlyOnce()
    {
        var services = TestServices.Create();
        var logger = new LifecycleRecordingLogger();
        var coordinator = new AssistantRuntimeLifecycleCoordinator(
            new ModuleLifecyclePlan([AssistantLifecycleParticipant.BargeIn]),
            AssistantOwnedPlan("BargeIn.Module"),
            new DuplexAssistantAudioFrontend(services.Audio),
            wakeup: null,
            speaker: null,
            bargeIn: services.PlaybackDetector,
            new AssistantTaskSupervisor(logger),
            logger);

        coordinator.DisposeAssistantOwned();
        Assert.Equal(1, services.PlaybackDetector.DisposeCount);
        coordinator.DisposeAssistantOwned();

        Assert.Equal(1, services.PlaybackDetector.DisposeCount);
        Assert.Throws<ObjectDisposedException>(() => { _ = services.PlaybackDetector.GetSnapshot(); });
    }

    [Fact]
    public async Task LifecycleCoordinator_IgnoresDisposedBargeInDuringStopWithoutLoggingFailure()
    {
        var services = TestServices.Create();
        var logger = new LifecycleRecordingLogger();
        var coordinator = new AssistantRuntimeLifecycleCoordinator(
            new ModuleLifecyclePlan([AssistantLifecycleParticipant.BargeIn]),
            AssistantOwnedPlan("BargeIn.Module"),
            new DuplexAssistantAudioFrontend(services.Audio),
            wakeup: null,
            speaker: null,
            services.PlaybackDetector,
            new AssistantTaskSupervisor(logger),
            logger);
        coordinator.DisposeAssistantOwned();

        await coordinator.StopAsync();

        Assert.Empty(logger.Entries);
    }

    [Fact]
    public async Task LifecycleCoordinator_LogsNonTerminalParticipantStopFailure()
    {
        var services = TestServices.Create();
        services.Audio.StopException = new InvalidOperationException("audio stop failure");
        var logger = new LifecycleRecordingLogger();
        var coordinator = new AssistantRuntimeLifecycleCoordinator(
            new ModuleLifecyclePlan([AssistantLifecycleParticipant.AudioFrontend]),
            AssistantOwnedPlan("AudioFrontend.Adapter"),
            new DuplexAssistantAudioFrontend(services.Audio),
            wakeup: null,
            speaker: null,
            bargeIn: null,
            new AssistantTaskSupervisor(logger),
            logger);

        await coordinator.StopAsync();

        Assert.Equal([LogLevel.Warning], logger.Entries);
    }

    /// <summary>Ownership plan marking one participant AssistantOwned for disposal tests.</summary>
    private static ModuleOwnership[] AssistantOwnedPlan(string moduleId)
        => [new ModuleOwnership(moduleId, ResourceOwnership.AssistantOwned)];

    private static void IgnoreProcessedCapture(in ProcessedCaptureFrame frame)
    {
    }

    private static void IgnoreDuplexAudio(in DuplexAudioFrame frame)
    {
    }

    private sealed class LifecycleRecordingLogger : ILogger
    {
        private readonly List<LogLevel> _levels = [];

        public IReadOnlyList<LogLevel> Entries
        {
            get
            {
                lock (_levels)
                    return _levels.ToArray();
            }
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(
            LogLevel logLevel,
            EventId eventId,
            TState state,
            Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            lock (_levels)
                _levels.Add(logLevel);
        }
    }
}
