using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Runtime.Control.Controllers;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Read-only view of the committed Turn snapshot. Collaborators and capability modules classify
/// facts against this view; <see cref="AssistantControlLoop"/> stays the only committer.
/// </summary>
internal interface ITurnStateView
{
    TurnSnapshot Snapshot { get; }

    AssistantState ConversationState { get; }

    long TurnId { get; }

    long NextTurnOrdinal { get; }

    bool IsCurrentTurn(long turnId);
}
