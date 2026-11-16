namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>
/// Plays the calibration speech used for controlled double-talk enrollment through the same
/// playback queue the assistant uses, at the normal product gain. The application owns the TTS or
/// recorded stimulus; the enrollment coordinator only starts and stops it and checks that render is
/// actually active. Tests use a fake.
/// </summary>
public interface ISpeakerEnrollmentStimulusPlayer
{
    /// <summary>Starts (or restarts) the calibration playback; returns once audio has been queued.</summary>
    Task StartAsync(CancellationToken cancellationToken);

    /// <summary>Stops the calibration playback and clears its queued audio.</summary>
    Task StopAsync(CancellationToken cancellationToken);
}
