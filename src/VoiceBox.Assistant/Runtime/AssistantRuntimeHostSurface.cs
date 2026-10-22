using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime;

/// <summary>
/// Read-only public projection owned by the Shell's publication/state collaborators. Keeping this
/// surface separate leaves AssistantRuntime as a lifecycle facade instead of a feature-state bag.
/// </summary>
internal sealed class AssistantRuntimeHostSurface
{
    private readonly AssistantRuntimeLifecycle _lifecycle;
    private readonly AssistantEventDispatcher _events;
    private readonly ITurnStateView _turnState;
    private readonly SpeechBoundaryIngress _speechBoundary;
    private readonly TurnContextStore _turnContexts;
    private readonly RecognitionTurnCoordinator _recognition;
    private readonly IWakeupModule? _wakeup;

    public AssistantRuntimeHostSurface(
        AssistantRuntimeLifecycle lifecycle,
        AssistantEventDispatcher events,
        ITurnStateView turnState,
        SpeechBoundaryIngress speechBoundary,
        TurnContextStore turnContexts,
        RecognitionTurnCoordinator recognition,
        IWakeupModule? wakeup)
    {
        _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
        _events = events ?? throw new ArgumentNullException(nameof(events));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _speechBoundary = speechBoundary ?? throw new ArgumentNullException(nameof(speechBoundary));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _wakeup = wakeup;
    }

    public AssistantState State
        => _lifecycle.State is AssistantRuntimeLifecycleState.Stopped or AssistantRuntimeLifecycleState.Disposed
            ? AssistantState.Stopped
            : _turnState.ConversationState == AssistantState.Idle &&
              _speechBoundary.HasPendingListeningProjection
                ? AssistantState.Listening
                : _turnState.ConversationState;

    public AssistantWakeMode WakeMode => _wakeup?.Mode ?? AssistantWakeMode.Disabled;

    public int ActiveTurnContextCount => _turnContexts.Count;

    public int DeferredRecognitionReadyTurnCount => _recognition.DeferredCount;

    internal void BindSender(object sender) => _events.BindSender(sender);

    public event EventHandler<AssistantStateChangedEventArgs>? StateChanged
    {
        add => _events.StateChanged += value;
        remove => _events.StateChanged -= value;
    }

    public event EventHandler<AssistantWakeModeChangedEventArgs>? WakeModeChanged
    {
        add => _events.WakeModeChanged += value;
        remove => _events.WakeModeChanged -= value;
    }

    public event EventHandler<AssistantWakeupDetectedEventArgs>? WakeupDetected
    {
        add => _events.WakeupDetected += value;
        remove => _events.WakeupDetected -= value;
    }

    public event EventHandler<UserSpokeEventArgs>? UserSpoke
    {
        add => _events.UserSpoke += value;
        remove => _events.UserSpoke -= value;
    }

    public event EventHandler<BargeInValidationEventArgs>? BargeInValidation
    {
        add => _events.BargeInValidation += value;
        remove => _events.BargeInValidation -= value;
    }

    public event EventHandler<AssistantFaultedEventArgs>? Faulted
    {
        add => _events.Faulted += value;
        remove => _events.Faulted -= value;
    }

    public event EventHandler<AssistantTextEventArgs>? AssistantText
    {
        add => _events.AssistantText += value;
        remove => _events.AssistantText -= value;
    }

    public event EventHandler<AssistantResponseGeneratedEventArgs>? AssistantResponseGenerated
    {
        add => _events.AssistantResponseGenerated += value;
        remove => _events.AssistantResponseGenerated -= value;
    }

    public event EventHandler<AssistantReplyEventArgs>? AssistantReplied
    {
        add => _events.AssistantReplied += value;
        remove => _events.AssistantReplied -= value;
    }
}
