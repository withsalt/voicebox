using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend.AudioDeviceEnumtor
{
    internal class DefaultHostApiAudioDeviceEnumtor : AudioDeviceEnumtorBase, IAudioDeviceEnumtor
    {
        public IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType)
        {
            if (_deviceCount <= 0)
            {
                return [];
            }

            (int targetApi, HostApiInfo? targetHostApi) = GetTargetApi();
            if (targetApi < 0 || targetHostApi == null)
            {
                throw new InvalidOperationException("Unable to determine target API name for audio devices.");
            }

            // 默认设备必须用目标 host API 自己的默认值判定:
            // 全局 Pa_GetDefaultInputDevice/Pa_GetDefaultOutputDevice 属于 PortAudio 默认 host API
            // (Windows 上是 MME), 与 WASAPI 设备索引永远对不上。
            int defaultInputDevice = targetHostApi.Value.defaultInputDevice;
            int defaultOutputDevice = targetHostApi.Value.defaultOutputDevice;

            int targetHostApiDeviceCount = 0;
            List<AudioDevice> devices = new List<AudioDevice>(_deviceCount);
            for (int i = 0; i < _deviceCount; i++)
            {
                //当枚举玩当前hostapi所有设备后，就跳出
                if (targetHostApiDeviceCount >= targetHostApi.Value.deviceCount)
                {
                    break;
                }

                DeviceInfo? deviceInfoNullable = Backend.PortAudioBackend.GetDeviceInfo(i);
                if (deviceInfoNullable == null)
                {
                    continue;
                }
                DeviceInfo deviceInfo = deviceInfoNullable.Value;

                if (deviceInfo.hostApi < 0 || deviceInfo.hostApi != targetApi)
                {
                    continue;
                }

                if (deviceInfo.maxInputChannels <= 0 && deviceInfo.maxOutputChannels <= 0)
                {
                    continue;
                }

                targetHostApiDeviceCount++;

                var currentDeviceType = GetAudioDeviceType(deviceInfo);
                bool isDefault = (currentDeviceType == AudioDeviceType.Input || currentDeviceType == AudioDeviceType.InputOutput) && i == defaultInputDevice
                    || (currentDeviceType == AudioDeviceType.Output || currentDeviceType == AudioDeviceType.InputOutput) && i == defaultOutputDevice;

                if (deviceType == null)
                {
                    devices.Add(new AudioDevice(i, deviceInfo.name, isDefault, deviceInfo.maxInputChannels, deviceInfo.maxOutputChannels, deviceInfo.defaultSampleRate, currentDeviceType, targetHostApi.Value.type));
                }
                else
                {
                    if (!MatchesDeviceType(currentDeviceType, deviceType))
                    {
                        continue;
                    }
                    devices.Add(new AudioDevice(i, deviceInfo.name, isDefault, deviceInfo.maxInputChannels, deviceInfo.maxOutputChannels, deviceInfo.defaultSampleRate, currentDeviceType, targetHostApi.Value.type));
                }
            }
            return devices;
        }

        protected (int, HostApiInfo?) GetTargetApi()
        {
            // Windows 上优先 WASAPI: MME 的设备名被系统截断为 31 字符 (WAVEINCAPS.szPname), 且延迟高。
            (int preferredApi, HostApiInfo? preferredInfo) = Backend.PortAudioBackend.GetPreferredHostApi();
            if (preferredApi >= 0 && preferredInfo != null)
            {
                return (preferredApi, preferredInfo);
            }

            // 其他平台 (或 WASAPI 不可用): 跟随默认设备所在的 host API
            int defaultInputDevice = Backend.PortAudioBackend.DefaultInputDevice;
            int defaultOutputDevice = Backend.PortAudioBackend.DefaultOutputDevice;
            int targetDevice = defaultInputDevice >= 0 ? defaultInputDevice : defaultOutputDevice >= 0 ? defaultOutputDevice : -1;
            if (targetDevice < 0)
                return (-1, null);

            DeviceInfo? deviceInfoNullable = Backend.PortAudioBackend.GetDeviceInfo(targetDevice);
            if (deviceInfoNullable == null)
                return (-1, null);
            DeviceInfo deviceInfo = deviceInfoNullable.Value;
            HostApiInfo? deviceHostApiInfo = Backend.PortAudioBackend.GetHostApiInfo(deviceInfo.hostApi);
            if (deviceHostApiInfo == null)
                return (-1, null);
            return (deviceInfo.hostApi, deviceHostApiInfo);
        }
    }
}
