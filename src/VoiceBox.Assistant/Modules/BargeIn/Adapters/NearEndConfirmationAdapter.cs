using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Confirmation-transaction facet of the near-end data plane.</summary>
internal sealed class NearEndConfirmationAdapter : INearEndConfirmationStore
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndConfirmationAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot()
        => _dataPlane.GetDiagnosticSnapshot();

    public bool TryAcquire(long operationId, out NearEndConfirmationLease lease)
        => _dataPlane.TryAcquire(operationId, out lease);

    public bool TryCommit(in NearEndConfirmationLease lease, in SpeakingDecision confirmed)
        => _dataPlane.TryCommit(in lease, in confirmed);

    public void Abort(in NearEndConfirmationLease lease)
        => _dataPlane.Abort(in lease);
}
