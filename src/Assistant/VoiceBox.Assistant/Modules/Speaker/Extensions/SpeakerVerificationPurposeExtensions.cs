using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Extensions;

internal static class SpeakerVerificationPurposeExtensions
{
    public static SpeakerPurposeBank Bank(this SpeakerVerificationPurpose purpose)
        => purpose switch
        {
            SpeakerVerificationPurpose.BargeInCommand => SpeakerPurposeBank.DoubleTalkNearEnd,
            _ => SpeakerPurposeBank.CleanNearEnd,
        };
}
