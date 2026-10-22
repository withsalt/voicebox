using VoiceBox.Audio.PortAudio.Backend;
using VoiceBox.Audio.PortAudio.Backend.AudioDeviceEnumtor;
using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio
{
    public abstract class PortAudioServiceBase : IAudioService
    {
        protected bool _disposed = false;
        protected volatile AudioServiceState _state = AudioServiceState.Idle;
        protected readonly SemaphoreSlim _asyncLock = new SemaphoreSlim(1, 1);
        private bool _portAudioReferenceAcquired = false;

        /// <summary>
        /// 获取当前服务状态
        /// </summary>
        public AudioServiceState State => _state;

        /// <summary>
        /// 当前音频服务配置。
        /// </summary>
        public abstract AudioOptionsBase Options { get; }

        public PortAudioServiceBase()
        {

        }

        protected void AcquirePortAudioReference()
        {
            if (_portAudioReferenceAcquired)
                return;

            if (!PortAudioManager.Instance.AcquireReference())
            {
                throw new ObjectDisposedException(nameof(PortAudioManager));
            }

            _portAudioReferenceAcquired = true;
        }

        public IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType = null, bool enumAllDevices = false)
        {
            CheckDisposed();
            IAudioDeviceEnumtor deviceEnumtor = enumAllDevices ? new AllHostApiAudioDeviceEnumtor() : new DefaultHostApiAudioDeviceEnumtor();
            return deviceEnumtor.ListAudioDevices(deviceType);
        }

        protected virtual void OptionsValidator(AudioOptionsBase options)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            // Validate sample rate
            if (options.SampleRate < 8000 || options.SampleRate > 192000)
                throw new ArgumentOutOfRangeException(nameof(options.SampleRate), $"Sample rate must be between 8000-192000 Hz. Current value: {options.SampleRate}");

            // Validate channel count
            if (options.ChannelCount < 1 || options.ChannelCount > 8)
                throw new ArgumentOutOfRangeException(nameof(options.ChannelCount), $"Channel count must be between 1-8. Current value: {options.ChannelCount}");
        }

        internal DeviceInfo GetAudioDevice(ref int deviceIndex, AudioDeviceType audioDeviceType)
        {
            if (deviceIndex == -1)
            {
                if (audioDeviceType != AudioDeviceType.Input && audioDeviceType != AudioDeviceType.Output)
                {
                    throw new InvalidOperationException("Audio type only support input and output..");
                }

                // Windows 上优先 WASAPI host API 的默认设备 (设备名完整、延迟低);
                // 其他平台 (或 WASAPI 不可用) 等价于 Pa_GetDefaultInputDevice/Pa_GetDefaultOutputDevice。
                deviceIndex = audioDeviceType == AudioDeviceType.Input ? PortAudioBackend.PreferredDefaultInputDevice : PortAudioBackend.PreferredDefaultOutputDevice;
                if (deviceIndex == PortAudioBackend.NoDevice)
                {
                    throw new InvalidOperationException("No default input device found.");
                }
            }

            DeviceInfo? deviceInfo = PortAudioBackend.GetDeviceInfo(deviceIndex);
            if (deviceInfo == null)
            {
                throw new InvalidOperationException($"Device with index {deviceIndex} not found.");
            }
            return deviceInfo.Value;
        }

        public virtual void Dispose()
        {
            if (_disposed)
                return;

            try
            {
                _asyncLock?.Dispose();
                if (_portAudioReferenceAcquired)
                {
                    PortAudioManager.Instance.ReleaseReference();
                    _portAudioReferenceAcquired = false;
                }
            }
            finally
            {
                _disposed = true;
            }
        }

        /// <summary>
        /// 检查对象是否已被释放。
        /// </summary>
        protected void CheckDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(GetType().Name);
            }
        }
    }
}
