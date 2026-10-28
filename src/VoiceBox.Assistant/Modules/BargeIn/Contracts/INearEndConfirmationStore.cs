using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Explicit confirmation transaction shared with the control-loop confirmation handler. A lease
/// carries the operation identity; commit and abort accept only the current lease, so a late or
/// duplicate control decision cannot mutate a newer capture.
/// </summary>
internal interface INearEndConfirmationStore
{
    ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot();

    bool TryAcquire(long operationId, out NearEndConfirmationLease lease);

    /// <summary>
    /// Starts capture only while the lease still owns the reply being interrupted. Returns false
    /// when reset, a playback boundary or an owner change invalidated the transaction.
    /// </summary>
    bool TryCommit(
        in NearEndConfirmationLease lease,
        in SpeakingDecision confirmed);

    void Abort(in NearEndConfirmationLease lease);
}
