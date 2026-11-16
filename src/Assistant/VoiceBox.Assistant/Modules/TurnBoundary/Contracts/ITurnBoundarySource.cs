using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

/// <summary>
/// Typed turn-boundary port. The Runtime consumes speech onset/audio/endpoint facts through this
/// boundary, so replacing the boundary source (VAD today, push-to-talk or a host endpoint later)
/// requires no Runtime or Kernel change.
/// </summary>
internal interface ITurnBoundarySource
{
    event EventHandler<EventArgs>? SpeechStarted;

    event EventHandler<TurnBoundarySpeechEventArgs>? Speaking;

    event EventHandler<TurnBoundarySpeechEventArgs>? SpeechEnded;

    /// <summary>Drops any partially accumulated boundary state.</summary>
    void Reset();

    /// <summary>Submits one processed capture block to the boundary detector.</summary>
    void Observe(ReadOnlySpan<float> audio);
}
