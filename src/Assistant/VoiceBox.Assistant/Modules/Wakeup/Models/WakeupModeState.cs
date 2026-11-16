using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

/// <summary>
/// Wake mode state and the continuous-conversation idle clock. Instances are mutated only while
/// the owning <see cref="WakeupSessionGate"/> holds its single state lock, which keeps a mode
/// change atomic while mode ownership lives in one place.
/// </summary>
internal sealed class WakeupModeState
{
    private readonly TimeProvider _timeProvider;
    private long _lastActivityTimestamp;

    public WakeupModeState(TimeProvider timeProvider)
    {
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
        _lastActivityTimestamp = _timeProvider.GetTimestamp();
    }

    public AssistantWakeMode Mode { get; private set; } = AssistantWakeMode.WaitingForWakeWord;

    public bool IsListeningForWakeWordOrInSession
        => Mode is AssistantWakeMode.WaitingForWakeWord or AssistantWakeMode.ContinuousConversation;

    public void ResetToWaitingForWakeWord()
    {
        Mode = AssistantWakeMode.WaitingForWakeWord;
        TouchActivity();
    }

    /// <summary>Enters continuous conversation and reports the previous mode.</summary>
    public AssistantWakeMode EnterConversation()
    {
        AssistantWakeMode previous = Mode;
        Mode = AssistantWakeMode.ContinuousConversation;
        return previous;
    }

    /// <summary>Leaves continuous conversation; returns false when already waiting.</summary>
    public bool TryLeaveConversation(out AssistantWakeMode previous)
    {
        previous = Mode;
        if (Mode != AssistantWakeMode.ContinuousConversation)
            return false;

        Mode = AssistantWakeMode.WaitingForWakeWord;
        return true;
    }

    public bool IsIdleTimeoutElapsed(TimeSpan idleTimeout)
        => Mode == AssistantWakeMode.ContinuousConversation &&
            _timeProvider.GetElapsedTime(_lastActivityTimestamp, _timeProvider.GetTimestamp()) >= idleTimeout;

    public void TouchActivity() => _lastActivityTimestamp = _timeProvider.GetTimestamp();

    public void TouchActivityIfInConversation()
    {
        if (Mode == AssistantWakeMode.ContinuousConversation)
            TouchActivity();
    }
}
