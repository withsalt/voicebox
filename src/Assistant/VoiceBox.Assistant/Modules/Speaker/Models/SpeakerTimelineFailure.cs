using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal enum SpeakerTimelineFailure
{
    None,
    NoStream,
    StreamEpochMismatch,
    TimelineGap,
    AudioExpired,
    NotYetAvailable,
    RangeInvalid,
}
