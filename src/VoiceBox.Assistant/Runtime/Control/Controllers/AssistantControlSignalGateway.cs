using System.Diagnostics;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>
/// The single admission point through which capability modules hand correlated semantic signals
/// to the control loop. It owns no Turn state and takes no decision: it stamps identity, applies
/// the correlation gate, admits to the bounded ingress and awaits the consumer's verdict.
/// </summary>
internal sealed class AssistantControlSignalGateway :
    IAssistantControlSignalSink,
    IResponseControlSignalPort,
    IRecognitionControlSignalPort,
    IBargeInValidationControlSignalPort
{
    private readonly AssistantControlIngress _ingress;
    private readonly ControlOperationRegistry _operations;
    private readonly OperationCorrelationGate _correlation;
    private readonly ControlPlaneAvailability _availability;
    private readonly ITurnStateView _turnState;
    private readonly TurnContextStore _turnContexts;
    private readonly InputGate _inputGate;
    private readonly RuntimeRunScope _runScope;

    public AssistantControlSignalGateway(
        AssistantControlIngress ingress,
        ControlOperationRegistry operations,
        OperationCorrelationGate correlation,
        ControlPlaneAvailability availability,
        ITurnStateView turnState,
        TurnContextStore turnContexts,
        InputGate inputGate,
        RuntimeRunScope runScope)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _correlation = correlation ?? throw new ArgumentNullException(nameof(correlation));
        _availability = availability ?? throw new ArgumentNullException(nameof(availability));
        _turnState = turnState ?? throw new ArgumentNullException(nameof(turnState));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _inputGate = inputGate ?? throw new ArgumentNullException(nameof(inputGate));
        _runScope = runScope ?? throw new ArgumentNullException(nameof(runScope));
    }

    public async ValueTask SubmitAsync(
        AssistantSignal signal,
        AssistantSignalSource source,
        CancellationToken cancellationToken)
        => await EnqueueAsync(signal, source, cancellationToken).ConfigureAwait(false);

    /// <summary>Allocates an identity for a semantic decision already on the control consumer.</summary>
    internal long NextOperationId() => _operations.NextOperationId();

    /// <summary>
    /// Admits a lifecycle command as an ordered barrier and waits for the consumer to reach it.
    /// </summary>
    public async Task EnqueueCommandAsync(
        RuntimeCommandKind kind,
        CancellationToken cancellationToken = default)
    {
        long operationId = _operations.NextOperationId();
        TaskCompletionSource completion = _operations.RegisterCommand(operationId);
        try
        {
            await _ingress.EnqueueAsync(
                new RuntimeCommand(kind, operationId),
                Stopwatch.GetTimestamp(),
                cancellationToken).ConfigureAwait(false);
            await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _operations.ForgetCommand(operationId);
        }
    }

    public async ValueTask<AssistantControlSignalResult> EnqueueAsync(
        AssistantSignal signal,
        AssistantSignalSource source,
        CancellationToken cancellationToken)
    {
        if (!_availability.IsAcceptingSignals)
            throw new OperationCanceledException("Assistant control signal was rejected during shutdown.");

        if (!_correlation.TryAdmit(signal.TurnId, signal.OperationId, signal.Kind))
            return new AssistantControlSignalResult(false, false, 0);

        TaskCompletionSource<AssistantControlSignalResult> completion =
            _operations.RegisterSignal(signal);

        try
        {
            long sourceTimestamp = Stopwatch.GetTimestamp();
            if (signal.Kind == AssistantSignalKind.InterruptionResolved)
            {
                // Validation resolution is an interruption decision, not a generic completion.
                // Keep it on the bounded critical lane so queue admission cannot wait behind
                // ordinary recognition/response completions; the control loop still merges both
                // lanes by IngressSequence.
                if (!_ingress.TryEnqueueCritical(signal, sourceTimestamp, source, out _))
                {
                    throw new InvalidOperationException(
                        "Assistant interruption decision was rejected by control ingress.");
                }
            }
            else
            {
                await _ingress.EnqueueSignalAsync(
                    signal,
                    sourceTimestamp,
                    source,
                    cancellationToken).ConfigureAwait(false);
            }

            return await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            // The producer is no longer waiting after success, cancellation or an admission
            // failure. Release only this exact registration: a later stage may legitimately reuse
            // the semantic operation id with a different signal kind before this finally runs.
            _operations.ForgetSignal(signal, completion);
        }
    }

    bool IResponseControlSignalPort.IsCurrentTurn(long turnId) => _turnState.IsCurrentTurn(turnId);

    void IResponseControlSignalPort.OpenInputForBargeIn(long turnId)
    {
        if (_turnState.IsCurrentTurn(turnId) && _runScope.CanOpenInput)
            _inputGate.Open();
    }

    public async ValueTask PublishRecognitionCompletedAsync(
        long turnId,
        string text,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        long operationId = _operations.NextOperationId();
        await EnqueueAsync(
            AssistantSignal.RecognitionCompleted(turnId, operationId, text, speechDuration),
            AssistantSignalSource.Recognition,
            cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask<ResponseControlTransition> StartResponseAsync(
        long turnId,
        long operationId,
        CancellationToken cancellationToken)
    {
        AssistantControlSignalResult result = await EnqueueAsync(
            AssistantSignal.ResponseStarted(turnId, operationId),
            AssistantSignalSource.Response,
            cancellationToken).ConfigureAwait(false);
        return new ResponseControlTransition(result.Accepted, result.Changed, result.PublicationSequence);
    }

    public async ValueTask<ResponseControlTransition> StartSpeakableResponseAsync(
        long turnId,
        long operationId,
        CancellationToken cancellationToken)
    {
        AssistantControlSignalResult result = await EnqueueAsync(
            AssistantSignal.ResponseSpeakableStarted(turnId, operationId),
            AssistantSignalSource.Response,
            cancellationToken).ConfigureAwait(false);
        return new ResponseControlTransition(result.Accepted, result.Changed, result.PublicationSequence);
    }

    public async ValueTask PublishResponseCompletedAsync(
        long turnId,
        long operationId,
        string text,
        CancellationToken cancellationToken)
        => await EnqueueAsync(
            AssistantSignal.ResponseCompleted(turnId, operationId, text),
            AssistantSignalSource.Response,
            cancellationToken).ConfigureAwait(false);

    public async ValueTask PublishPlaybackCompletedAsync(
        long turnId,
        long operationId,
        string text,
        TimeSpan elapsed,
        CancellationToken cancellationToken)
        => await EnqueueAsync(
            AssistantSignal.PlaybackCompleted(turnId, operationId, text, elapsed),
            AssistantSignalSource.Playback,
            cancellationToken).ConfigureAwait(false);

    public async ValueTask PublishResponseTerminatedAsync(
        long turnId,
        long operationId,
        ResponseOperationOutcomeKind outcome,
        string? failure,
        CancellationToken cancellationToken)
    {
        AssistantSignal signal = outcome == ResponseOperationOutcomeKind.Cancelled
            ? AssistantSignal.OperationCancelled(turnId, operationId)
            : AssistantSignal.OperationFailed(
                AssistantSignalKind.ResponseFailed,
                turnId,
                operationId,
                new AssistantFailure(
                    "VBX-RESP-001",
                    "response",
                    "ResponseModule",
                    failure ?? "response-failed"));
        await EnqueueAsync(signal, AssistantSignalSource.Response, cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask<bool> PublishRecognitionStartedAsync(
        long turnId,
        long operationId,
        string sessionId,
        CancellationToken cancellationToken)
    {
        AssistantControlSignalResult result = await EnqueueAsync(
            AssistantSignal.RecognitionStarted(turnId, operationId, sessionId),
            AssistantSignalSource.Recognition,
            cancellationToken).ConfigureAwait(false);
        return result.Accepted;
    }

    public async ValueTask PublishRecognitionFailedAsync(
        RecognitionOutcome outcome,
        CancellationToken cancellationToken)
        => await EnqueueAsync(
            AssistantSignal.OperationFailed(
                AssistantSignalKind.RecognitionFailed,
                outcome.TurnId,
                outcome.OperationId,
                outcome.Failure),
            AssistantSignalSource.Recognition,
            cancellationToken).ConfigureAwait(false);

    public async ValueTask<BargeInValidationControlResolution> ResolveValidationAsync(
        long turnId,
        string? text,
        string reason,
        bool meaningful,
        CancellationToken cancellationToken)
    {
        long operationId = _operations.NextOperationId();
        AssistantControlSignalResult result = await EnqueueAsync(
            AssistantSignal.ValidationResolved(turnId, operationId, text, reason, meaningful),
            AssistantSignalSource.Validation,
            cancellationToken).ConfigureAwait(false);
        return new BargeInValidationControlResolution(result.Accepted);
    }

    public async ValueTask PublishValidatedRecognitionCompletedAsync(
        long turnId,
        string text,
        string? rawText,
        TimeSpan speechDuration,
        CancellationToken cancellationToken)
    {
        _turnContexts.TryCreate(turnId);
        _turnContexts.TrySetRecognitionMetadata(turnId, rawText, speechDuration);
        long operationId = _operations.NextOperationId();
        await EnqueueAsync(
            AssistantSignal.RecognitionCompleted(turnId, operationId, text, speechDuration),
            AssistantSignalSource.Validation,
            cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask PublishValidationCancelledAsync(
        long turnId,
        CancellationToken cancellationToken)
    {
        long operationId = _operations.NextOperationId();
        await EnqueueAsync(
            AssistantSignal.ValidationCancelled(turnId, operationId),
            AssistantSignalSource.Validation,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Admits a wake-mode transition on the critical lane. Losing it would strand the public
    /// wake-mode contract, so an admission failure converges as a terminal control fault.
    /// </summary>
    public void SubmitWakeModeChanged(AssistantWakeMode from, AssistantWakeMode to)
    {
        if (from == to)
            return;

        long operationId = _operations.NextOperationId();
        if (!_ingress.TryEnqueueCritical(
                AssistantSignal.WakeupModeChanged(operationId, from, to),
                Stopwatch.GetTimestamp(),
                AssistantSignalSource.Wakeup,
                out _) &&
            !_ingress.HasTerminalFault)
        {
            _ingress.TryTerminate(
                new AssistantFailure(
                    "VBX-CTRL-019",
                    "capacity",
                    "wakeup-control-plane",
                    "wake-mode-admission-failed"),
                Stopwatch.GetTimestamp(),
                out _);
        }
    }
}
