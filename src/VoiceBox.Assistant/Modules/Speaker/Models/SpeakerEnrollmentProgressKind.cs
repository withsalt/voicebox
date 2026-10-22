namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>What a progress notification reports, so UIs can react without inferring it from fields.</summary>
public enum SpeakerEnrollmentProgressKind
{
    /// <summary>A stage boundary (preparing, building, saving, completed, failed).</summary>
    StageChanged = 0,
    /// <summary>A segment attempt is about to record; the user should start speaking.</summary>
    SegmentStarted = 1,
    /// <summary>Speech is accumulating for the current segment attempt.</summary>
    Collecting = 2,
    /// <summary>The current segment attempt was accepted or rejected (see <see cref="SpeakerEnrollmentProgressEventArgs.ReasonCode"/>).</summary>
    SegmentOutcome = 3,
    /// <summary>A later <c>EnrollAsync</c> is continuing from a previous failed segment on the same coordinator.</summary>
    Resumed = 4,
}
