using MiniAudioEx.Native;

namespace VoiceBox.Audio.MiniAudio.Backend;

internal sealed class MiniAudioDeviceSession : IDisposable
{
    private readonly ma_device_data_proc _dataCallback;
    private ma_context_ptr _context;
    private ma_device_ptr _device;
    private bool _contextInitialized;
    private bool _deviceInitialized;
    private bool _started;
    private bool _disposed;

    internal MiniAudioDeviceSession(
        MiniAudioDeviceSelection? playbackDevice,
        MiniAudioDeviceSelection? captureDevice,
        uint sampleRate,
        uint channels,
        int maxFrameSize,
        double? suggestedLatency,
        bool enableWasapiExclusiveMode,
        ma_device_data_proc dataCallback)
    {
        if (playbackDevice is null && captureDevice is null)
        {
            throw new ArgumentException("At least one audio endpoint is required.");
        }

        ma_backend backend = playbackDevice?.Backend ?? captureDevice!.Value.Backend;
        if (playbackDevice.HasValue && captureDevice.HasValue && playbackDevice.Value.Backend != captureDevice.Value.Backend)
        {
            throw new InvalidOperationException("A duplex device must use playback and capture endpoints from the same MiniAudio backend.");
        }

        _dataCallback = dataCallback ?? throw new ArgumentNullException(nameof(dataCallback));
        MiniAudioNativeRuntime.ContextGate.Wait();
        try
        {
            _context = new ma_context_ptr(true);
            if (_context.pointer == IntPtr.Zero)
            {
                throw new OutOfMemoryException("Could not allocate a MiniAudio context.");
            }

            ma_result contextResult = MiniAudioNative.ma_context_init([backend], _context);
            MiniAudioNativeRuntime.ThrowIfFailed(contextResult, "context initialization");
            _contextInitialized = true;

            _device = new ma_device_ptr(true);
            if (_device.pointer == IntPtr.Zero)
            {
                throw new OutOfMemoryException("Could not allocate a MiniAudio device.");
            }

            ma_device_type deviceType = playbackDevice.HasValue && captureDevice.HasValue
                ? ma_device_type.duplex
                : playbackDevice.HasValue
                    ? ma_device_type.playback
                    : ma_device_type.capture;
            ma_device_config config = MiniAudioNative.ma_device_config_init(deviceType);
            config.sampleRate = sampleRate;
            config.noFixedSizedCallback = 0;
            if (maxFrameSize > 0)
            {
                config.periodSizeInFrames = (uint)maxFrameSize;
            }
            else if (suggestedLatency.HasValue || maxFrameSize == -1)
            {
                config.periodSizeInMilliseconds = (uint)Math.Clamp(
                    Math.Round((suggestedLatency ?? 0.01) * 1000),
                    1,
                    1000);
            }

            config.SetDataCallback(_dataCallback);
            ma_share_mode shareMode = enableWasapiExclusiveMode && backend == ma_backend.wasapi
                ? ma_share_mode.exclusive
                : ma_share_mode.shared;

            ma_device_id_ptr playbackId = default;
            ma_device_id_ptr captureId = default;
            try
            {
                if (playbackDevice.HasValue)
                {
                    playbackId = AllocateDeviceId(playbackDevice.Value.PlaybackId);
                    config.playback.pDeviceID = playbackId;
                    config.playback.format = ma_format.f32;
                    config.playback.channels = channels;
                    config.playback.shareMode = shareMode;
                }

                if (captureDevice.HasValue)
                {
                    captureId = AllocateDeviceId(captureDevice.Value.CaptureId);
                    config.capture.pDeviceID = captureId;
                    config.capture.format = ma_format.f32;
                    config.capture.channels = channels;
                    config.capture.shareMode = shareMode;
                }

                ma_result deviceResult = MiniAudioNative.ma_device_init(_context, ref config, _device);
                MiniAudioNativeRuntime.ThrowIfFailed(deviceResult, "device initialization");
                _deviceInitialized = true;
            }
            finally
            {
                captureId.Free();
                playbackId.Free();
            }
        }
        catch
        {
            DisposeCore();
            throw;
        }
        finally
        {
            MiniAudioNativeRuntime.ContextGate.Release();
        }
    }

    internal void Start()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_started)
        {
            return;
        }

        MiniAudioNativeRuntime.ThrowIfFailed(MiniAudioNative.ma_device_start(_device), "device start");
        _started = true;
    }

    internal void Stop()
    {
        if (_disposed || !_started)
        {
            return;
        }

        MiniAudioNativeRuntime.ThrowIfFailed(MiniAudioNative.ma_device_stop(_device), "device stop");
        _started = false;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        MiniAudioNativeRuntime.ContextGate.Wait();
        try
        {
            DisposeCore();
        }
        finally
        {
            MiniAudioNativeRuntime.ContextGate.Release();
        }
    }

    private static unsafe ma_device_id_ptr AllocateDeviceId(ma_device_id? deviceId)
    {
        if (!deviceId.HasValue)
        {
            throw new InvalidOperationException("The selected endpoint has no native device identifier.");
        }

        ma_device_id_ptr pointer = new(true);
        if (pointer.pointer == IntPtr.Zero)
        {
            throw new OutOfMemoryException("Could not allocate a MiniAudio device identifier.");
        }

        *pointer.Get() = deviceId.Value;
        return pointer;
    }

    private void DisposeCore()
    {
        if (_deviceInitialized)
        {
            if (_started)
            {
                MiniAudioNative.ma_device_stop(_device);
                _started = false;
            }

            MiniAudioNative.ma_device_uninit(_device);
            _deviceInitialized = false;
        }

        if (_device.pointer != IntPtr.Zero)
        {
            _device.Free();
        }

        if (_contextInitialized)
        {
            MiniAudioNative.ma_context_uninit(_context);
            _contextInitialized = false;
        }

        if (_context.pointer != IntPtr.Zero)
        {
            _context.Free();
        }

        _disposed = true;
    }
}
