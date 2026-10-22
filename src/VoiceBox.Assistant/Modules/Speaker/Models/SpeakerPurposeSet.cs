using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Effective enabled purposes after composition applied the KWS gate.</summary>
internal readonly record struct SpeakerPurposeSet(bool Wakeup, bool BargeInCommand, bool SingleTalk)
{
    public static SpeakerPurposeSet None => default;

    public bool Any => Wakeup || BargeInCommand || SingleTalk;
}
