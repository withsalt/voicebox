using System.Diagnostics;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// Consumes admitted near-end signals on the control loop. Unlike the data-plane sink this type
/// can safely depend on confirmation, recognition and speech-boundary owners.
/// </summary>
internal sealed class ActiveNearEndControlHandler
{
    private readonly AssistantControlIngress _ingress;
    private readonly INearEndConfirmationStore _confirmations;
    private readonly BargeInTurnCoordinator _bargeInTurns;
    private readonly IRecognitionAudioPort _recognition;
    private readonly ITurnBoundaryEndpointSink _speechBoundary;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;

    public ActiveNearEndControlHandler(
        AssistantControlIngress ingress,
        INearEndConfirmationStore confirmations,
        BargeInTurnCoordinator bargeInTurns,
        IRecognitionAudioPort recognition,
        ITurnBoundaryEndpointSink speechBoundary,
        VoiceAssistantBenchmarkProbe? benchmarkProbe)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _confirmations = confirmations ?? throw new ArgumentNullException(nameof(confirmations));
        _bargeInTurns = bargeInTurns ?? throw new ArgumentNullException(nameof(bargeInTurns));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _speechBoundary = speechBoundary ?? throw new ArgumentNullException(nameof(speechBoundary));
        _benchmarkProbe = benchmarkProbe;
    }

    /// <summary>
    /// Resolves a near-end confirmation on the control consumer: it becomes a barge-in only if the
    /// current snapshot still allows one.
    /// </summary>
    public BargeInSemanticPlan ProcessInterruptionRequested(long operationId)
    {
        _benchmarkProbe?.ObserveControlTrace("near-end.confirmation.reduce.begin", operationId: operationId);
        ActiveNearEndConfirmationRequest request = default;
        SpeakingDecision confirmed = default;
        bool tookConfirmation = _confirmations.TryTakeConfirmation(operationId, out request);
        // A KWS-enabled candidate reaches the control plane only after its own speaker Match; a
        // KWS-disabled one only after an acoustic Interrupt. Both are re-checked against the current
        // snapshot here, and the stop itself is issued by the Kernel, never by this handler.
        bool identityPermits = !request.IdentityRequired || request.IdentityAuthorized;
        bool confirmedNearEnd = tookConfirmation &&
            identityPermits &&
            _bargeInTurns.TryConfirmFromNearEndWindow(
                request.InitialAudio,
                request.Decision,
                request.IdentityRequired && request.IdentityAuthorized,
                out confirmed);
        if (confirmedNearEnd)
        {
            _benchmarkProbe?.ObserveControlTrace(
                "near-end.confirmation.reduced",
                confirmed.TurnId,
                operationId);
            _confirmations.ApplyConfirmedCapture(request, confirmed);
            ObserveBenchmarkProgress();
            return _bargeInTurns.ObserveConfirmedBargeIn(
                confirmed,
                BargeInTriggerSource.NearEndWindow,
                request.InitialAudio,
                request.IdentityRequired
                    ? BargeInIdentityContext.AuthorizedSpeaker(request.CandidateId.Value)
                    : BargeInIdentityContext.NotRequired);
        }

        if (tookConfirmation)
            _confirmations.DiscardTakenConfirmation(operationId);
        ObserveBenchmarkProgress();
        return default;
    }

    /// <summary>
    /// Drains the near-end and recognition feeds before reporting the capture end, so the endpoint
    /// never finalises a session ahead of its final PCM block.
    /// </summary>
    public async Task DrainSpeechAsync(long turnId, TimeSpan duration, CancellationToken cancellationToken)
    {
        try
        {
            await _recognition.DrainAudioAsync();
            cancellationToken.ThrowIfCancellationRequested();
            if (!_ingress.TryEnqueueCritical(
                    AssistantSignal.ActiveNearEndSpeechDrainCompleted(turnId, duration),
                    Stopwatch.GetTimestamp(),
                    AssistantSignalSource.DuplexCallback,
                    out _) &&
                !_ingress.HasTerminalFault)
            {
                Terminate("VBX-CTRL-018", "speech-drain-completion-admission-failed");
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
    }

    public SpeechBoundaryObservation ProcessSpeechDrainCompleted(long turnId, TimeSpan duration)
    {
        // The interruption is accepted at authorization time while the data-plane drain may still
        // be completing. Only a turn that still owns its interruption context may be closed by the
        // capture-end fact.
        if (_bargeInTurns.IsValidating(turnId))
            return _speechBoundary.HandleSpeechEnded(duration);

        return default;
    }

    private void ObserveBenchmarkProgress()
        => _benchmarkProbe?.ObserveControlPlaneProgress(
            _confirmations.GetDiagnosticSnapshot(),
            _bargeInTurns.ConversationState,
            _bargeInTurns.IsSpeakingWithoutPendingCandidate);

    private void Terminate(string code, string messageKey)
        => _ingress.TryTerminate(
            new AssistantFailure(code, "capacity", "active-near-end-data-plane", messageKey),
            Stopwatch.GetTimestamp(),
            out _);
}
