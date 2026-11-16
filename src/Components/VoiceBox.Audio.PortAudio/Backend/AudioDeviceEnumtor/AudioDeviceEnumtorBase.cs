using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend.AudioDeviceEnumtor
{
    internal class AudioDeviceEnumtorBase
    {
        protected int _deviceCount;

        public AudioDeviceEnumtorBase()
        {
            _deviceCount = Backend.PortAudioBackend.DeviceCount;
        }

        protected AudioDeviceType GetAudioDeviceType(DeviceInfo deviceInfo)
        {
            if (deviceInfo.maxInputChannels > 0 && deviceInfo.maxOutputChannels > 0)
            {
                return AudioDeviceType.InputOutput;
            }
            else if (deviceInfo.maxInputChannels > 0)
            {
                return AudioDeviceType.Input;
            }
            else if (deviceInfo.maxOutputChannels > 0)
            {
                return AudioDeviceType.Output;
            }
            else
            {
                return AudioDeviceType.Unknown;
            }
        }

        protected bool MatchesDeviceType(AudioDeviceType currentDeviceType, AudioDeviceType? requestedDeviceType)
        {
            if (requestedDeviceType == null)
                return currentDeviceType != AudioDeviceType.Unknown;

            if (currentDeviceType == AudioDeviceType.Unknown)
                return false;

            AudioDeviceType requested = requestedDeviceType.Value;
            if (requested == AudioDeviceType.InputOutput)
                return currentDeviceType == AudioDeviceType.InputOutput;

            bool requestsInput = (requested & AudioDeviceType.Input) != 0;
            bool requestsOutput = (requested & AudioDeviceType.Output) != 0;

            return currentDeviceType switch
            {
                AudioDeviceType.Input => requestsInput,
                AudioDeviceType.Output => requestsOutput,
                AudioDeviceType.InputOutput => requestsInput || requestsOutput,
                _ => false
            };
        }
    }
}
