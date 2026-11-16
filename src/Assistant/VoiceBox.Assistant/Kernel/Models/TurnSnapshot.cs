using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Kernel.Models;

internal sealed record TurnSnapshot
{
    public AssistantState ConversationState { get; init; } = AssistantState.Idle;
    public AssistantWakeMode WakeMode { get; init; } = AssistantWakeMode.Disabled;
    public long TurnId { get; init; }
    public long NextTurnOrdinal { get; init; } = 1;
    public long InterruptedTurnId { get; init; }
    public long ValidationTurnId { get; init; }
    public bool InputGateOpen { get; init; } = true;
    public ResponseLifecycle ResponseLifecycle { get; init; }

    /// <summary>Identity authorization of the interruption candidate identified by <see cref="ValidationTurnId"/>.</summary>
    public TurnAuthorizationStatus Authorization { get; init; }

    /// <summary>Speaker candidate correlated with the pending interruption; 0 when none.</summary>
    public long CandidateId { get; init; }

    /// <summary>Whether the interrupted turn's playback stop has already been issued for this candidate.</summary>
    public bool PhysicalStopIssued { get; init; }

    public PhysicalStopTrigger PhysicalStopTrigger { get; init; }

    public static TurnSnapshot Initial { get; } = new();
}
