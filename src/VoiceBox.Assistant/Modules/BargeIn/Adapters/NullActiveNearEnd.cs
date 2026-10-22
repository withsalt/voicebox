using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// Typed near-end data-plane boundary. The implementation owns candidate/capture/session state and
/// bounded audio reservoirs; Runtime only supplies observations and consumes low-frequency outcomes.
/// </summary>
internal sealed class NullActiveNearEnd :
    INearEndCapturePort,
    INearEndRenderPort,
    INearEndEndpointPort,
    INearEndConfirmationStore,
    ISpeechTimelineLocator,
    INearEndCaptureControlPort,
    INearEndResetPort
{
    public static NullActiveNearEnd Instance { get; } = new();

    private NullActiveNearEnd()
    {
    }

    public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot() => default;

    public ActiveNearEndProcessingState GetProcessingState() => default;

    public void ObserveEndpointSpeechStarted() { }

    public void ObserveEndpointSpeechAudio(ReadOnlySpan<float> audio) { }

    public void ObserveUnownedEndpointSpeechEnded() { }

    public void ObserveEndpointSpeechEnded(bool captureCommittedAfterEndpoint = false) { }

    public void ObserveDuplexAudio(
        in DuplexAudioFrame frame,
        bool playbackStartupGuardActive,
        bool latestNearEndSpeech = false) { }

    public MainAudioObservation ObserveRecognitionAudio(in ProcessedCaptureFrame frame) => default;

    public void FeedFromRecognitionAudio(ReadOnlySpan<float> audio, MainAudioObservation observation) { }

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
    {
        streamEpoch = 0;
        startSample = 0;
        endSample = 0;
        return false;
    }

    public bool StopCapture(long turnId, bool startCooldown) => false;

    public void Reset() { }

    public void ClearAudioReservoirs() { }

    public bool TryTakeConfirmation(long operationId, out ActiveNearEndConfirmationRequest request)
    {
        request = default;
        return false;
    }

    public void DiscardTakenConfirmation(long operationId) { }

    public void ApplyConfirmedCapture(
        in ActiveNearEndConfirmationRequest request,
        in SpeakingDecision confirmed) { }

}
