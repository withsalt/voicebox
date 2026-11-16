using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    internal enum NearendVadNotificationKind
    {
        SpeechStarted,
        Speaking,
        SpeechEnded,
    }

    internal readonly record struct NearendVadNotification(
        NearendVadNotificationKind Kind,
        float[] Audio,
        VADEventDataType DataType,
        TimeSpan? Duration = null)
    {
        public static NearendVadNotification SpeechStarted { get; } =
            new(NearendVadNotificationKind.SpeechStarted, Array.Empty<float>(), VADEventDataType.None);
    }
}
