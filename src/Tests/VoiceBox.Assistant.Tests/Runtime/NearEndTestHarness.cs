using VoiceBox.Assistant.Modules.BargeIn.Adapters;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Typed seam over a composed near-end module. Tests drive the same ports the runtime uses and
/// observe state through the owners the module exposes, never through reflection.
/// </summary>
internal sealed class NearEndTestHarness
{
    private readonly NearEndModule _module;

    public NearEndTestHarness(NearEndModule module, IRecognitionAudioPort recognitionAudio)
    {
        _module = module ?? throw new ArgumentNullException(nameof(module));
        RecognitionAudio = recognitionAudio ?? throw new ArgumentNullException(nameof(recognitionAudio));
    }

    public NearEndModule Module => _module;

    public NearEndPorts Ports => _module.Ports;

    public NearEndDataPlane DataPlane => _module.DataPlane;

    public IRecognitionAudioPort RecognitionAudio { get; }

    public INearEndConfirmationStore Confirmations => Ports.Confirmation;

    public long CommandAuthorizedCount => DataPlane.CommandAuthorizedCount;

    public long CommandDeniedCount => DataPlane.CommandDeniedCount;

    public MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame)
        => Ports.Capture.ObserveRecognitionAudio(in frame);

    public void FeedFromRecognitionAudio(ReadOnlySpan<float> audio, MainAudioObservation observation)
        => Ports.Capture.FeedFromRecognitionAudio(audio, observation);

    public void ObserveDuplexAudio(
        in DuplexAudioFrame frame,
        bool playbackStartupGuardActive,
        bool latestNearEndSpeech = false)
        => Ports.Render.ObserveDuplexAudio(in frame, playbackStartupGuardActive, latestNearEndSpeech);

    public void ObserveEndpointSpeechStarted() => Ports.Endpoint.ObserveEndpointSpeechStarted();

    public void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio)
        => Ports.Endpoint.ObserveEndpointSpeechAudio(audio);

    public void ObserveUnownedEndpointSpeechEnded()
        => Ports.Endpoint.ObserveUnownedEndpointSpeechEnded();

    public void ObserveEndpointSpeechEnded(bool captureCommittedAfterEndpoint = false)
        => Ports.Endpoint.ObserveEndpointSpeechEnded(captureCommittedAfterEndpoint);

    public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot()
        => Ports.Render.GetDiagnosticSnapshot();

    public ActiveNearEndProcessingState GetProcessingState()
        => Ports.Render.GetProcessingState();

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
        => Ports.Timeline.TryLocate(audio, out streamEpoch, out startSample, out endSample);

    public bool StopCapture(long turnId, bool startCooldown)
        => Ports.CaptureControl.StopCapture(turnId, startCooldown);

    public void Reset() => Ports.Reset.Reset();

    public void ClearAudioReservoirs() => Ports.Reset.ClearAudioReservoirs();

    public bool TryAcquire(long operationId, out NearEndConfirmationLease lease)
        => Confirmations.TryAcquire(operationId, out lease);

    public void ApplyConfirmedCapture(in NearEndConfirmationLease lease, in SpeakingDecision confirmed)
    {
        // Most data-plane tests commit a hand-built transaction without exercising the control
        // loop. Acquire it here when the journal does not already hold the same lease; a
        // different taken lease is a test bug, not a silent no-op.
        if (_module.Journal.HasTaken)
        {
            if (!_module.Journal.TryGetReplayRequest(out NearEndConfirmationLease taken) ||
                taken.OperationId != lease.OperationId ||
                taken.CandidateId != lease.CandidateId)
            {
                throw new InvalidOperationException(
                    "A different confirmation lease is already taken.");
            }
        }
        else if (!_module.Journal.TryAcquire(lease.OperationId, out _) &&
                 (!_module.Journal.TryQueue(in lease) ||
                  !_module.Journal.TryAcquire(lease.OperationId, out _)))
        {
            throw new InvalidOperationException("The confirmation lease could not be acquired.");
        }

        if (!Confirmations.TryCommit(in lease, in confirmed))
            throw new InvalidOperationException("The confirmation lease could not be committed.");
    }

    public void DiscardTakenConfirmation(in NearEndConfirmationLease lease)
        => Confirmations.Abort(in lease);

    /// <summary>
    /// Seeds the confirmation transaction directly on its owner. This is explicit typed setup of
    /// a transaction the control plane would normally acquire, not reflection over private state.
    /// </summary>
    public void SeedPendingConfirmation(in NearEndConfirmationLease lease)
    {
        if (!_module.Journal.TryQueue(in lease))
            throw new InvalidOperationException("A confirmation is already queued.");
        _module.Replay.ResetNextSample(lease.Selection.SelectedEndSample);
    }
}
