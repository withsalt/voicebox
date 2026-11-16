using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Narrow read-only view of the playback barge-in validation turn. Other modules use it to tell a
/// validation turn apart from the active turn without holding the validation owner's state.
/// </summary>
internal interface IBargeInValidationTurnView
{
    /// <summary>
    /// Atomically captures the values needed by the realtime recognition path. Callers must take
    /// this snapshot before entering another owner's lock; the cancellation source is borrowed
    /// and remains owned by the validation turn.
    /// </summary>
    BargeInValidationCaptureSnapshot CaptureSnapshot { get; }

    /// <summary>0 when no playback interruption is being validated.</summary>
    long ValidationTurnId { get; }

    /// <summary>The turn a pending validation would interrupt, or 0.</summary>
    long InterruptedTurnId { get; }

    /// <summary>
    /// The response cancellation source that belongs to the validation context, when present.
    /// Capture routing exposes it only as the semantic cancellation target of a validation feed;
    /// ownership remains with the validation owner.
    /// </summary>
    CancellationTokenSource? ResponseCts { get; }

    bool IsValidating(long turnId);
}
