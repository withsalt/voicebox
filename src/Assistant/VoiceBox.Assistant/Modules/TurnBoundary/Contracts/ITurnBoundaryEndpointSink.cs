using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

/// <summary>Reports a speech endpoint to the turn-boundary coordinator.</summary>
internal interface ITurnBoundaryEndpointSink
{
    SpeechBoundaryObservation HandleSpeechEnded(TimeSpan duration);
}
