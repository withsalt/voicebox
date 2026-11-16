using VoiceBox.Assistant.Modules.AudioFrontend.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Contracts;

/// <summary>Composition-time factory for the optional speaker module.</summary>
internal interface ISpeakerModuleFactory
{
    ISpeakerModule Create(
        SpeakerCaptureDescriptor descriptor,
        int audioSampleRate,
        int audioChannels,
        Microsoft.Extensions.Logging.ILogger logger);
}
