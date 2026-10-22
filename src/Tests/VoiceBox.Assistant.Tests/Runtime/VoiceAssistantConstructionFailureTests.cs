using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Composition.Wiring.Models;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public async Task PlaybackWorkerFault_StopStillReleasesAudioAndSubscriptions()
    {
        var services = TestServices.Create();
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var failure = new InvalidOperationException("device clear failed");
        services.Audio.ClearQueueException = failure;
        using var assistant = services.CreateAssistant();
        using var cancellation = new CancellationTokenSource();
        await assistant.StartAsync();
        var submission = await assistant.EnqueueSpeechAsync("speech", cancellation.Token);
        await WaitUntilAsync(() => services.Audio.AcceptedSamples > 0);
        cancellation.Cancel();
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => submission.Completion.WaitAsync(TimeSpan.FromSeconds(2))));
        Assert.Same(failure, await Assert.ThrowsAsync<InvalidOperationException>(
            () => assistant.StopAsync().WaitAsync(TimeSpan.FromSeconds(2))));
        Assert.Equal(1, services.Audio.StopCount);
        Assert.False(services.Audio.IsRecording);
        Assert.Equal(AssistantState.Stopped, assistant.State);
        var run = RuntimeGraphProbe.RequireField<RuntimeRunScope>(assistant, "_runScope");
        Assert.False(run.IsStarted);
        Assert.Null(typeof(FakeDuplexService)
            .GetField("OnProcessedCaptureFrame", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .GetValue(services.Audio));
    }

    [Fact]
    public async Task PublicationWorkerFault_StillStopsModulesAndRetiresRunScope()
    {
        var services = TestServices.Create();
        using var assistant = services.CreateAssistant();
        int faultCount = 0;
        assistant.Faulted += (_, _) => Interlocked.Increment(ref faultCount);
        await assistant.StartAsync();
        var boundary = RuntimeGraphProbe.RequireField<OrderedPublicationBoundary>(assistant, "_publications");
        var run = RuntimeGraphProbe.RequireField<RuntimeRunScope>(assistant, "_runScope");
        boundary.ReserveSequence();
        boundary.RaiseAssistantText(new AssistantTextEventArgs { Delta = "gap", IsThinking = false });

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => boundary.Completion.WaitAsync(TimeSpan.FromSeconds(2)));
        await WaitUntilAsync(() => services.Audio.StopCount == 1);
        InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => assistant.StopAsync().WaitAsync(TimeSpan.FromSeconds(2)));
        Assert.Contains("gap", failure.Message, StringComparison.Ordinal);
        Assert.Equal(1, Volatile.Read(ref faultCount));
        Assert.Equal(AssistantState.Stopped, assistant.State);
        Assert.False(run.IsStarted);
        Assert.False(services.Audio.IsRecording);
    }

    [Fact]
    public void CompositionFailure_AfterPublicationConstructionDoesNotRootEventSender()
    {
        WeakReference sender = FailCompositionAfterPublicationConstruction();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.False(sender.IsAlive, "A failed composition retained its sender through a publication thread.");
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference FailCompositionAfterPublicationConstruction()
    {
        var services = TestServices.Create();
        var composition = VoiceAssistant.CreateBuilder()
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();
        AssistantRuntimePlan original = composition.RuntimePlan;
        var failure = new InvalidOperationException("format provider failed");
        var plan = new AssistantRuntimePlan(
            new FailingCompositionFrontend(original.Audio, failure),
            original.TurnBoundary, original.Recognition, original.ResponseGeneration,
            original.SpeechSynthesis, original.Wakeup, original.Speaker,
            original.AcousticDiagnostics, original.PlaybackBargeInDetectorFactory);
        var installed = new AssistantInstalledModules(
            null, null, null, plan.Recognition, plan.ResponseGeneration.Create());
        var eventSender = new object();
        var weakSender = new WeakReference(eventSender);
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
            AssistantRuntimeCompositionFactory.Create(
                eventSender, plan, composition.Lifecycle, installed,
                Options.Create(new VoiceAssistantOptions()), NullLogger.Instance,
                TimeProvider.System, null)));
        return weakSender;
    }

    private sealed class FailingCompositionFrontend(IAssistantAudioFrontend frontend, Exception failure)
        : IAssistantAudioFrontend
    {
        public IAssistantAudioInput Input { get; } = new FailingFormatInput(failure);
        public IAssistantAudioOutput Output => frontend.Output;
        public IProcessedCaptureSource? ProcessedCapture => frontend.ProcessedCapture;
        public IRenderReferenceSource? RenderReference => frontend.RenderReference;
        public IDuplexAudioFrameSource? DuplexFrames => frontend.DuplexFrames;
        public SpeakerCaptureDescriptor? SpeakerCapture => frontend.SpeakerCapture;
        public bool SupportsSimultaneousCapturePlayback => frontend.SupportsSimultaneousCapturePlayback;
        public bool SupportsRenderReference => frontend.SupportsRenderReference;
        public bool SupportsProcessedCaptureTimeline => frontend.SupportsProcessedCaptureTimeline;
        public bool SupportsSpeakerCaptureTimeline => frontend.SupportsSpeakerCaptureTimeline;
        public Task StartAsync(CancellationToken cancellationToken = default) => frontend.StartAsync(cancellationToken);
        public Task StopAsync(CancellationToken cancellationToken = default) => frontend.StopAsync(cancellationToken);
    }

    private sealed class FailingFormatInput(Exception failure) : IAssistantAudioInput
    {
        public uint SampleRate => throw failure;
        public uint ChannelCount => 1;
        public SampleFormat SampleFormat => SampleFormat.Float32;
        public bool IsCapturing => false;
    }
}
