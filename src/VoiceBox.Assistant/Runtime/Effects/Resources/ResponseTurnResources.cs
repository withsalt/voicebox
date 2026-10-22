using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.TurnContext;
using VoiceBox.Assistant.Runtime.TurnContext.Contracts;

namespace VoiceBox.Assistant.Runtime.Effects.Resources;

/// <summary>
/// Lends a response operation the runtime-owned resources it needs — the run token, an operation
/// identity and the turn's cancellation source — and takes them back exactly once. It decides no
/// semantic state; turn completion is submitted as a correlated signal like any other.
/// </summary>
internal sealed class ResponseTurnResources : IResponseTurnResourcePort
{
    private readonly TurnContextStore _turnContexts;
    private readonly ITurnResourcePort _resources;
    private readonly ITurnStateView _turnState;
    private readonly ControlOperationRegistry _operations;
    private readonly ControlPlaneAvailability _availability;
    private readonly RuntimeRunScope _runScope;
    private AssistantControlSignalGateway? _controlSignals;

    public ResponseTurnResources(
        TurnContextStore turnContexts,
        ITurnResourcePort resources,
        ITurnStateView turnState,
        ControlOperationRegistry operations,
        ControlPlaneAvailability availability,
        RuntimeRunScope runScope)
    {
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _resources = resources ?? throw new ArgumentNullException(nameof(resources));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _availability = availability ?? throw new ArgumentNullException(nameof(availability));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
    }

    /// <summary>Composition binds the gateway once the control plane exists.</summary>
    public void Bind(AssistantControlSignalGateway controlSignals)
        => _controlSignals = controlSignals ?? throw new ArgumentNullException(nameof(controlSignals));

    public CancellationToken RunToken => _runScope.Token;

    public bool IsShuttingDown => _availability.IsShuttingDown;

    public bool TryAttachResponseContext(long turnId, CancellationTokenSource responseCts)
    {
        if (!_turnContexts.TrySetResponseCts(turnId, responseCts))
            return false;

        if (!_turnState.IsCurrentTurn(turnId))
            return false;

        _resources.GetOrCreate(turnId).ResponseCts = responseCts;
        return true;
    }

    public void ReleaseResponseContext(long turnId, CancellationTokenSource responseCts)
    {
        ArgumentNullException.ThrowIfNull(responseCts);
        TurnResourceSlot slot = _resources.GetOrCreate(turnId);
        if (_turnState.IsCurrentTurn(turnId) && ReferenceEquals(slot.ResponseCts, responseCts))
            slot.ResponseCts = null;

        if (_turnContexts.TryTakeResponseCts(turnId, out CancellationTokenSource? owned))
            owned!.Dispose();
        else
            responseCts.Dispose();

        _turnContexts.Release(turnId);
    }

    public async ValueTask PublishTurnCompletedAsync(long turnId, bool endConversationRequested)
    {
        long operationId = _operations.NextOperationId();
        await _controlSignals!.EnqueueAsync(
            AssistantSignal.ValidationTurnCompleted(turnId, operationId, endConversationRequested),
            AssistantSignalSource.Response,
            CancellationToken.None).ConfigureAwait(false);
    }

    /// <summary>Cancels the active turn's response without disposing a source it does not own.</summary>
}
