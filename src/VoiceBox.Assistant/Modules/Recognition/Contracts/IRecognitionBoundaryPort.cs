using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>Receives unclaimed capture audio and endpoints and returns turn-boundary decisions.</summary>
internal interface IRecognitionBoundaryPort : IRecognitionTurnLifecyclePort
{
    SpeakingDecision ObserveAudio(ReadOnlySpan<float> audio);

    SpeechEndedDecision ObserveSpeechEnded(TimeSpan duration);
}
