namespace VoiceBox.Assistant.Kernel.Models;

/// <summary>What legitimately issued the single physical stop of the interrupted playback.</summary>
internal enum PhysicalStopTrigger
{
    None,

    /// <summary>KWS enabled: the current candidate's speaker Match.</summary>
    SpeakerMatch,

    /// <summary>KWS disabled: the double-talk detector reached Interrupt, or the basic VAD gate confirmed.</summary>
    AcousticInterrupt,
}
