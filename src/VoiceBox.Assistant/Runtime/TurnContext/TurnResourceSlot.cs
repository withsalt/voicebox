namespace VoiceBox.Assistant.Runtime.TurnContext;

/// <summary>
/// Runtime-owned mutable resources for one turn. The state machine may mutate a slot only while
/// holding its own decision lock; it never owns, disposes, or discovers the slot.
/// </summary>
internal sealed class TurnResourceSlot
{
    public CancellationTokenSource? ResponseCts { get; set; }
}
