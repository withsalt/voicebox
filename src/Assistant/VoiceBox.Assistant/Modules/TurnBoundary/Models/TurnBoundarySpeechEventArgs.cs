namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

/// <summary>
/// Protocol-neutral speech observation produced by a turn-boundary source. VAD-specific DTOs stay
/// inside the adapter and never leak into Runtime or Kernel.
/// </summary>
internal sealed class TurnBoundarySpeechEventArgs : EventArgs
{
    public TurnBoundarySpeechEventArgs(float[] audio, TimeSpan duration)
    {
        Audio = audio ?? [];
        Duration = duration;
    }

    public float[] Audio { get; }

    public TimeSpan Duration { get; }
}
