using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Confirmation ownership and its diagnostic snapshot, shared with the control-loop confirmation handler.
/// </summary>
internal interface INearEndConfirmationStore
{
    ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot();

    bool TryTakeConfirmation(long operationId, out ActiveNearEndConfirmationRequest request);

    void DiscardTakenConfirmation(long operationId);

    void ApplyConfirmedCapture(
        in ActiveNearEndConfirmationRequest request,
        in SpeakingDecision confirmed);
}
