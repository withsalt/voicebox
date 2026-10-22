using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend.AudioDeviceEnumtor
{
    internal interface IAudioDeviceEnumtor
    {
        IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType);
    }
}
