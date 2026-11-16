namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Acoustic condition an enrollment segment was recorded under.</summary>
public enum SpeakerEnrollmentCondition
{
    /// <summary>Owner speech with no assistant playback.</summary>
    Clean,

    /// <summary>Owner speech while calibration speech plays through the real output path (AEC engaged).</summary>
    DoubleTalk,
}
