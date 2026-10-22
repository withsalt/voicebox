using MiniAudioEx.Native;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio.Backend;

public abstract class MiniAudioServiceBase : IAudioService
{
    protected readonly SemaphoreSlim _asyncLock = new(1, 1);
    protected volatile AudioServiceState _state = AudioServiceState.Idle;
    protected bool _disposed;

    private MiniAudioDeviceCatalog? _catalog;

    public AudioServiceState State => _state;

    public abstract AudioOptionsBase Options { get; }

    public IReadOnlyList<AudioDevice> ListAudioDevices(
        AudioDeviceType? deviceType = null,
        bool enumAllDevices = false)
    {
        CheckDisposed();
        _catalog = MiniAudioDeviceCatalog.Enumerate(enumAllDevices);
        return _catalog.Devices
            .Select(selection => selection.AudioDevice)
            .Where(device => MatchesDeviceType(device.DeviceType, deviceType))
            .ToArray();
    }

    private protected MiniAudioDeviceSelection ResolveDevice(
        int deviceIndex,
        AudioDeviceType direction,
        ma_backend? preferredBackend = null)
    {
        CheckDisposed();
        MiniAudioDeviceCatalog catalog = _catalog ??= MiniAudioDeviceCatalog.Enumerate(false);
        if (deviceIndex < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        }

        if (deviceIndex >= 0)
        {
            MiniAudioDeviceSelection selected = catalog.Devices.FirstOrDefault(
                selection => selection.AudioDevice.Id == deviceIndex);
            if (selected.AudioDevice is null)
            {
                throw new InvalidOperationException($"Audio device index {deviceIndex} was not found in the current device list.");
            }

            EnsureDirection(selected, direction);
            return selected;
        }

        MiniAudioDeviceSelection? defaultDevice = catalog.Devices
            .Where(selection => SupportsDirection(selection, direction))
            .Where(selection => !preferredBackend.HasValue || selection.Backend == preferredBackend.Value)
            .FirstOrDefault(selection => selection.AudioDevice.IsDefault);
        if (defaultDevice.HasValue && defaultDevice.Value.AudioDevice is not null)
        {
            return defaultDevice.Value;
        }

        if (preferredBackend.HasValue)
        {
            MiniAudioDeviceSelection backendDefault = catalog.Devices.FirstOrDefault(
                selection => selection.Backend == preferredBackend.Value && SupportsDirection(selection, direction));
            if (backendDefault.AudioDevice is not null)
            {
                return backendDefault;
            }
        }

        throw new InvalidOperationException($"No default {direction.ToString().ToLowerInvariant()} device is available.");
    }

    protected int GetDefaultDeviceIndex(AudioDeviceType direction)
    {
        CheckDisposed();
        _catalog ??= MiniAudioDeviceCatalog.Enumerate(false);
        return _catalog.Devices.FirstOrDefault(selection =>
            SupportsDirection(selection, direction) && selection.AudioDevice.IsDefault).AudioDevice?.Id ?? -1;
    }

    protected void CheckDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, GetType().Name);
    }

    protected void ValidateOptions(AudioOptionsBase options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.SampleRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(nameof(options.SampleRate));
        }

        if (options.ChannelCount is < 1 or > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ChannelCount));
        }

        if (options.SampleFormat != SampleFormat.Float32)
        {
            throw new NotSupportedException("Only SampleFormat.Float32 is supported.");
        }
    }

    protected void ValidateBufferOptions(MiniAudioOptionsBase options)
    {
        ValidateOptions(options);
        if (options.MaxFrameSize < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(options.MaxFrameSize));
        }

        if (options.SuggestedLatency.HasValue &&
            (!double.IsFinite(options.SuggestedLatency.Value) ||
             options.SuggestedLatency.Value < 0.001 || options.SuggestedLatency.Value > 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(options.SuggestedLatency));
        }
    }

    public virtual void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _asyncLock.Dispose();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private static bool MatchesDeviceType(AudioDeviceType current, AudioDeviceType? requested)
    {
        if (requested is null)
        {
            return current != AudioDeviceType.Unknown;
        }

        if (requested.Value == AudioDeviceType.InputOutput)
        {
            return current == AudioDeviceType.InputOutput;
        }

        bool wantsInput = (requested.Value & AudioDeviceType.Input) != 0;
        bool wantsOutput = (requested.Value & AudioDeviceType.Output) != 0;
        return current switch
        {
            AudioDeviceType.Input => wantsInput,
            AudioDeviceType.Output => wantsOutput,
            AudioDeviceType.InputOutput => wantsInput || wantsOutput,
            _ => false,
        };
    }

    private static bool SupportsDirection(MiniAudioDeviceSelection selection, AudioDeviceType direction)
        => direction == AudioDeviceType.Input ? selection.SupportsCapture : selection.SupportsPlayback;

    private static void EnsureDirection(MiniAudioDeviceSelection selection, AudioDeviceType direction)
    {
        if (!SupportsDirection(selection, direction))
        {
            throw new InvalidOperationException($"Audio device {selection.AudioDevice.Id} does not support {direction.ToString().ToLowerInvariant()}.");
        }
    }
}
