using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;

namespace VoiceBox.Assistant.Runtime.Control.State;

/// <summary>
/// Holds the committed Turn snapshot of one Runtime instance. Reading is lock-free and open to
/// every collaborator; publishing a new snapshot is restricted to
/// <see cref="AssistantControlLoopTurnWriter"/>, which is driven by the single control consumer.
/// </summary>
internal sealed class TurnSnapshotStore : ITurnStateView
{
    private TurnSnapshot _snapshot = TurnSnapshot.Initial;
    private readonly object _changeLock = new();
    private TaskCompletionSource? _changed;

    public TurnSnapshot Snapshot => Volatile.Read(ref _snapshot);

    public AssistantState ConversationState => Snapshot.ConversationState;

    public long TurnId => Snapshot.TurnId;

    public long NextTurnOrdinal => Snapshot.NextTurnOrdinal;

    public bool IsCurrentTurn(long turnId) => Snapshot.TurnId == turnId;

    /// <summary>
    /// Publishes an accepted decision. Called only by the turn writer, after every private and
    /// data-plane projection has applied the same decision.
    /// </summary>
    internal void Publish(TurnSnapshot next)
    {
        TaskCompletionSource? changed;
        lock (_changeLock)
        {
            Volatile.Write(ref _snapshot, next);
            changed = _changed;
            _changed = null;
        }
        changed?.TrySetResult();
    }

    /// <summary>Waits without polling, including when a commit raced with the caller's read.</summary>
    internal Task WaitForChangeAsync(TurnSnapshot observed, CancellationToken cancellationToken)
    {
        lock (_changeLock)
        {
            if (!ReferenceEquals(observed, _snapshot))
                return Task.CompletedTask;

            _changed ??= new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            return _changed.Task.WaitAsync(cancellationToken);
        }
    }
}
