namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>
/// Why owner identity is being verified. <see cref="BargeInCommand"/> authorizes both the physical
/// stop and the semantic command of one playback-time candidate; there is deliberately no separate
/// "interrupt" purpose so one candidate can never carry two inconsistent identity results.
/// </summary>
internal enum SpeakerVerificationPurpose
{
    Wakeup,
    BargeInCommand,
    SingleTalk,
}
