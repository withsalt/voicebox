using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Explicit confirmation transaction: pending, taken (leased) and terminal commit/abort. Only the
/// current lease can commit or abort; late or duplicate operations leave newer candidates alone.
/// All mutation happens under the near-end data-plane lock.
/// </summary>
internal sealed class NearEndConfirmationJournal
{
    private NearEndConfirmationLease? _pending;
    private NearEndConfirmationLease? _taken;

    public bool HasPending => _pending.HasValue;

    public bool HasTaken => _taken.HasValue;

    public bool HasQueued => _pending.HasValue || _taken.HasValue;

    /// <summary>Queues one confirmation. At most one may be pending or taken at a time.</summary>
    public bool TryQueue(in NearEndConfirmationLease lease)
    {
        if (_pending.HasValue || _taken.HasValue)
            return false;

        _pending = lease;
        return true;
    }

    /// <summary>Moves the matching pending confirmation into the caller's lease.</summary>
    public bool TryAcquire(long operationId, out NearEndConfirmationLease lease)
    {
        if (_pending is { } pending && pending.OperationId == operationId)
        {
            _pending = null;
            _taken = pending;
            lease = pending;
            return true;
        }

        lease = default;
        return false;
    }

    /// <summary>The queued or leased request that confirmation-tail capture should follow.</summary>
    public bool TryGetReplayRequest(out NearEndConfirmationLease lease)
    {
        if (_pending is { } pending)
        {
            lease = pending;
            return true;
        }

        if (_taken is { } taken)
        {
            lease = taken;
            return true;
        }

        lease = default;
        return false;
    }

    public bool IsQueuedOperation(long operationId)
        => _pending?.OperationId == operationId || _taken?.OperationId == operationId;

    /// <summary>Discards a pending confirmation that could not be admitted to the control ingress.</summary>
    public bool TryDiscardPending(long operationId)
    {
        if (_pending is { } pending && pending.OperationId == operationId)
        {
            _pending = null;
            return true;
        }

        return false;
    }

    /// <summary>Commits exactly the current lease and releases it. Stale leases are ignored.</summary>
    public bool TryCommit(in NearEndConfirmationLease lease)
    {
        if (_taken is not { } taken ||
            taken.OperationId != lease.OperationId ||
            taken.CandidateId != lease.CandidateId)
        {
            return false;
        }

        _taken = null;
        return true;
    }

    /// <summary>Aborts exactly the current lease. Stale or repeated aborts are ignored.</summary>
    public bool TryAbort(in NearEndConfirmationLease lease)
    {
        if (_taken is not { } taken ||
            taken.OperationId != lease.OperationId ||
            taken.CandidateId != lease.CandidateId)
        {
            return false;
        }

        _taken = null;
        return true;
    }

    public void Clear()
    {
        _pending = null;
        _taken = null;
    }
}
