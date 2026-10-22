using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.Playback;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

namespace VoiceBox.Assistant.Runtime;

/// <summary>
/// Narrow facade over the AssistantRuntime Shell. It assembles nothing itself: Composition hands
/// it a finished collaborator set, and it only exposes the runtime lifecycle plus the read-only
/// projections the public API needs. Every conversation responsibility lives in a collaborator.
/// </summary>
internal sealed class AssistantRuntime : IDisposable
{
    private readonly AssistantRuntimeLifecycle _lifecycle;
    private readonly AssistantRuntimeHostSurface _hostSurface;
    private readonly ISpeechPlaybackControlPort _speechPlayback;

    public AssistantRuntime(AssistantRuntimeCompositionServices services)
    {
        ArgumentNullException.ThrowIfNull(services);
        _lifecycle = services.Lifecycle;
        _hostSurface = services.HostSurface;
        _speechPlayback = services.SpeechPlayback;
        _lifecycle.Bind(this);
    }

    public AssistantState State => _hostSurface.State;

    public AssistantWakeMode WakeMode => _hostSurface.WakeMode;

    internal int ActiveTurnContextCount => _hostSurface.ActiveTurnContextCount;

    internal int DeferredRecognitionReadyTurnCount => _hostSurface.DeferredRecognitionReadyTurnCount;

    public ValueTask<SpeechEnqueueResult> EnqueueSpeechAsync(
        string text,
        SpeechPlaybackQueuePosition queuePosition = SpeechPlaybackQueuePosition.Tail,
        CancellationToken cancellationToken = default)
        => _speechPlayback.EnqueueSpeechAsync(text, queuePosition, cancellationToken);

    public void StopCurrentSpeechPlayback()
        => _speechPlayback.StopCurrentSpeechPlayback();

    public event EventHandler<AssistantStateChangedEventArgs>? StateChanged
    {
        add => _hostSurface.StateChanged += value;
        remove => _hostSurface.StateChanged -= value;
    }

    public event EventHandler<AssistantWakeModeChangedEventArgs>? WakeModeChanged
    {
        add => _hostSurface.WakeModeChanged += value;
        remove => _hostSurface.WakeModeChanged -= value;
    }

    public event EventHandler<AssistantWakeupDetectedEventArgs>? WakeupDetected
    {
        add => _hostSurface.WakeupDetected += value;
        remove => _hostSurface.WakeupDetected -= value;
    }

    public event EventHandler<UserSpokeEventArgs>? UserSpoke
    {
        add => _hostSurface.UserSpoke += value;
        remove => _hostSurface.UserSpoke -= value;
    }

    public event EventHandler<BargeInValidationEventArgs>? BargeInValidation
    {
        add => _hostSurface.BargeInValidation += value;
        remove => _hostSurface.BargeInValidation -= value;
    }

    public event EventHandler<AssistantFaultedEventArgs>? Faulted
    {
        add => _hostSurface.Faulted += value;
        remove => _hostSurface.Faulted -= value;
    }

    public event EventHandler<AssistantTextEventArgs>? AssistantText
    {
        add => _hostSurface.AssistantText += value;
        remove => _hostSurface.AssistantText -= value;
    }

    public event EventHandler<AssistantResponseGeneratedEventArgs>? AssistantResponseGenerated
    {
        add => _hostSurface.AssistantResponseGenerated += value;
        remove => _hostSurface.AssistantResponseGenerated -= value;
    }

    public event EventHandler<AssistantReplyEventArgs>? AssistantReplied
    {
        add => _hostSurface.AssistantReplied += value;
        remove => _hostSurface.AssistantReplied -= value;
    }

    public Task StartAsync(CancellationToken ct = default) => _lifecycle.StartAsync(ct);

    public Task StopAsync() => _lifecycle.StopAsync();

    public void Dispose() => _lifecycle.Dispose();
}
