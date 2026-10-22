using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend.AudioDeviceEnumtor
{
    internal class AllHostApiAudioDeviceEnumtor : AudioDeviceEnumtorBase, IAudioDeviceEnumtor
    {
        public IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType)
        {
            if (_deviceCount <= 0)
            {
                return [];
            }

            int defaultInputDevice = Backend.PortAudioBackend.DefaultInputDevice;
            int defaultOutputDevice = Backend.PortAudioBackend.DefaultOutputDevice;

            Dictionary<int, HostApiInfo?> hostApiCache = new Dictionary<int, HostApiInfo?>();

            List<AudioDevice> devices = new List<AudioDevice>(_deviceCount);
            for (int i = 0; i < _deviceCount; i++)
            {
                DeviceInfo? deviceInfoNullable = Backend.PortAudioBackend.GetDeviceInfo(i);
                if (deviceInfoNullable == null)
                {
                    continue;
                }
                DeviceInfo deviceInfo = deviceInfoNullable.Value;
                if (deviceInfo.hostApi < 0)
                {
                    continue;
                }
                if (deviceInfo.maxInputChannels <= 0 && deviceInfo.maxOutputChannels <= 0)
                {
                    continue;
                }

                if (!hostApiCache.TryGetValue(deviceInfo.hostApi, out HostApiInfo? hostApiInfo))
                {
                    hostApiInfo = Backend.PortAudioBackend.GetHostApiInfo(deviceInfo.hostApi);
                    if (hostApiInfo != null)
                    {
                        hostApiCache[deviceInfo.hostApi] = hostApiInfo;
                    }
                }
                if (hostApiInfo == null)
                {
                    continue;
                }

                var currentDeviceType = GetAudioDeviceType(deviceInfo);
                bool isDefault = (currentDeviceType == AudioDeviceType.Input || currentDeviceType == AudioDeviceType.InputOutput) && i == defaultInputDevice
                    || (currentDeviceType == AudioDeviceType.Output || currentDeviceType == AudioDeviceType.InputOutput) && i == defaultOutputDevice;

                if (deviceType == null)
                {
                    devices.Add(new AudioDevice(i, deviceInfo.name, isDefault, deviceInfo.maxInputChannels, deviceInfo.maxOutputChannels, deviceInfo.defaultSampleRate, currentDeviceType, hostApiInfo.Value.type));
                }
                else
                {
                    if (!MatchesDeviceType(currentDeviceType, deviceType))
                    {
                        continue;
                    }
                    devices.Add(new AudioDevice(i, deviceInfo.name, isDefault, deviceInfo.maxInputChannels, deviceInfo.maxOutputChannels, deviceInfo.defaultSampleRate, currentDeviceType, hostApiInfo.Value.type));
                }
            }
            return devices;
        }
    }
}
