using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Gates;

/// <summary>
/// Runtime-side correlation ledger for asynchronous module completions. It records completion
/// identities before they reach the reducer, making duplicate and late observations diagnosable
/// without putting mutable operation state in the Kernel snapshot.
/// </summary>
internal sealed class OperationCorrelationGate
{
    private readonly object _lock = new();
    private readonly HashSet<OperationIdentity> _admitted = [];
    private readonly Dictionary<long, long> _latestTurnByOperation = [];
    private readonly Dictionary<long, HashSet<OperationIdentity>> _admissionsByTurn = [];
    private readonly Dictionary<long, HashSet<long>> _operationsByTurn = [];
    private readonly HashSet<long> _closedTurns = [];
    private readonly Queue<long> _closedTurnOrder = new();
    private const int RetainedClosedTurnCapacity = 4096;

    public bool TryAdmit(long turnId, long operationId, AssistantSignalKind signalKind)
    {
        // Completions without a current turn/operation identity are late or unknown and must not
        // enter the reducer. Lifecycle and mode signals may legally use turnId 0.
        if (turnId <= 0 || operationId <= 0)
            return !IsAsyncCompletion(signalKind);

        lock (_lock)
        {
            if (_closedTurns.Contains(turnId))
                return false;

            if (_latestTurnByOperation.TryGetValue(operationId, out long existingTurn) &&
                existingTurn != turnId)
            {
                return false;
            }

            _latestTurnByOperation[operationId] = turnId;
            var identity = new OperationIdentity(turnId, operationId, signalKind);
            if (!_admitted.Add(identity))
                return false;

            GetOrCreate(_admissionsByTurn, turnId).Add(identity);
            GetOrCreate(_operationsByTurn, turnId).Add(operationId);
            return true;
        }
    }

    internal int RetainedAdmissionCount
    {
        get { lock (_lock) return _admitted.Count; }
    }

    internal int RetainedOperationCount
    {
        get { lock (_lock) return _latestTurnByOperation.Count; }
    }

    private static bool IsAsyncCompletion(AssistantSignalKind signalKind)
        => signalKind is
            AssistantSignalKind.RecognitionStarted or
            AssistantSignalKind.RecognitionCompleted or
            AssistantSignalKind.RecognitionFailed or
            AssistantSignalKind.ResponseStarted or
            AssistantSignalKind.BackgroundResponseStarted or
            AssistantSignalKind.ResponseSpeakableStarted or
            AssistantSignalKind.ResponseCompleted or
            AssistantSignalKind.PlaybackCompleted or
            AssistantSignalKind.ResponseFailed or
            AssistantSignalKind.OperationCanceled or
            AssistantSignalKind.ValidationCancelled or
            AssistantSignalKind.TurnCompleted or
            AssistantSignalKind.InterruptionResolved;

    public void ForgetTurn(long turnId)
    {
        lock (_lock)
        {
            if (_closedTurns.Add(turnId))
            {
                _closedTurnOrder.Enqueue(turnId);
                while (_closedTurnOrder.Count > RetainedClosedTurnCapacity)
                {
                    long expiredTurnId = _closedTurnOrder.Dequeue();
                    _closedTurns.Remove(expiredTurnId);
                    ReleaseExpiredTurnCorrelations(expiredTurnId);
                }
            }
        }
    }

    private void ReleaseExpiredTurnCorrelations(long turnId)
    {
        if (_admissionsByTurn.Remove(turnId, out HashSet<OperationIdentity>? admissions))
        {
            foreach (OperationIdentity identity in admissions)
                _admitted.Remove(identity);
        }

        if (_operationsByTurn.Remove(turnId, out HashSet<long>? operationIds))
        {
            foreach (long operationId in operationIds)
            {
                if (_latestTurnByOperation.TryGetValue(operationId, out long ownerTurnId) &&
                    ownerTurnId == turnId)
                {
                    _latestTurnByOperation.Remove(operationId);
                }
            }
        }
    }

    private static HashSet<TValue> GetOrCreate<TValue>(
        Dictionary<long, HashSet<TValue>> entries,
        long turnId)
        where TValue : notnull
    {
        if (!entries.TryGetValue(turnId, out HashSet<TValue>? values))
        {
            values = [];
            entries.Add(turnId, values);
        }

        return values;
    }

    private readonly record struct OperationIdentity(
        long TurnId,
        long OperationId,
        AssistantSignalKind SignalKind);
}
