using System.Diagnostics;
using System.Text;
using MiniAudioEx.Native;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio.Backend;

internal sealed unsafe class MiniAudioDeviceCatalog
{
    private const int DeviceNameBufferSize = 256;

    private static readonly UTF8Encoding StrictUtf8 =
        new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private static readonly Lazy<Encoding?> AnsiEncoding = new(CreateAnsiEncoding);

    private readonly MiniAudioDeviceSelection[] _devices;

    private MiniAudioDeviceCatalog(MiniAudioDeviceSelection[] devices)
    {
        _devices = devices;
    }

    internal IReadOnlyList<MiniAudioDeviceSelection> Devices => _devices;

    internal static MiniAudioDeviceCatalog Enumerate(bool allBackends)
    {
        var builders = new List<DeviceBuilder>();
        ma_backend? defaultBackend = EnumerateContext(null, isDefaultContext: true, builders);

        if (allBackends)
        {
            foreach (ma_backend backend in Enum.GetValues<ma_backend>())
            {
                if (backend is ma_backend.custom or ma_backend.nill || backend == defaultBackend)
                {
                    continue;
                }

                EnumerateContext(backend, isDefaultContext: false, builders);
            }
        }

        var result = new MiniAudioDeviceSelection[builders.Count];
        for (int i = 0; i < builders.Count; i++)
        {
            DeviceBuilder builder = builders[i];
            AudioDevice audioDevice = new(
                i,
                builder.Name,
                builder.IsDefault,
                builder.MaxInputChannels,
                builder.MaxOutputChannels,
                builder.DefaultSampleRate,
                builder.DeviceType,
                MapHostApi(builder.Backend));
            result[i] = new MiniAudioDeviceSelection(
                audioDevice,
                builder.Backend,
                builder.CaptureId,
                builder.PlaybackId);
        }

        return new MiniAudioDeviceCatalog(result);
    }

    internal static HostApiType MapHostApi(ma_backend backend)
        => backend switch
        {
            ma_backend.wasapi => HostApiType.WASAPI,
            ma_backend.dsound => HostApiType.DirectSound,
            ma_backend.winmm => HostApiType.MME,
            ma_backend.coreaudio => HostApiType.CoreAudio,
            ma_backend.sndio => HostApiType.Sndio,
            ma_backend.audio4 => HostApiType.Audio4,
            ma_backend.oss => HostApiType.OSS,
            ma_backend.pulseaudio => HostApiType.PulseAudio,
            ma_backend.alsa => HostApiType.ALSA,
            ma_backend.jack => HostApiType.JACK,
            ma_backend.aaudio => HostApiType.AAudio,
            ma_backend.opensl => HostApiType.OpenSL,
            ma_backend.webaudio => HostApiType.WebAudio,
            _ => HostApiType.Other,
        };

    private static ma_backend? EnumerateContext(
        ma_backend? requestedBackend,
        bool isDefaultContext,
        List<DeviceBuilder> builders)
    {
        ma_context_ptr context = new(true);
        if (context.pointer == IntPtr.Zero)
        {
            throw new OutOfMemoryException("Could not allocate a MiniAudio context.");
        }

        bool initialized = false;
        MiniAudioNativeRuntime.ContextGate.Wait();
        try
        {
            ma_result initializeResult = requestedBackend.HasValue
                ? MiniAudioNative.ma_context_init([requestedBackend.Value], context)
                : MiniAudioNative.ma_context_init(null, context);
            if (initializeResult != ma_result.success)
            {
                return null;
            }

            initialized = true;
            ma_backend activeBackend = context.Get()->backend;
            if (activeBackend == ma_backend.nill)
            {
                return activeBackend;
            }

            ma_result devicesResult = MiniAudioNative.ma_context_get_devices(
                context,
                out ma_device_info[] playbackDevices,
                out ma_device_info[] captureDevices);
            if (devicesResult != ma_result.success)
            {
                return activeBackend;
            }

            foreach (ma_device_info info in playbackDevices ?? [])
            {
                AddDevice(builders, activeBackend, GetDetailedDeviceInfo(context, info, isCapture: false), isDefaultContext, isCapture: false);
            }

            foreach (ma_device_info info in captureDevices ?? [])
            {
                AddDevice(builders, activeBackend, GetDetailedDeviceInfo(context, info, isCapture: true), isDefaultContext, isCapture: true);
            }

            return activeBackend;
        }
        finally
        {
            if (initialized)
            {
                MiniAudioNative.ma_context_uninit(context);
            }

            context.Free();
            MiniAudioNativeRuntime.ContextGate.Release();
        }
    }

    private static void AddDevice(
        List<DeviceBuilder> builders,
        ma_backend backend,
        ma_device_info info,
        bool isDefaultContext,
        bool isCapture)
    {
        string name = GetDeviceName(info, backend);
        if (name.Length == 0)
        {
            return;
        }

        bool isDefault = isDefaultContext && info.isDefault > 0;
        // WinMM capture and playback indices belong to separate namespaces.
        // Other shared IDs can be combined only when their default flags agree;
        // AudioDevice has one IsDefault flag for both directions.
        DeviceBuilder? existing = backend == ma_backend.winmm ? null : builders.FirstOrDefault(builder =>
            builder.Backend == backend && builder.IsDefault == isDefault && DeviceIdsEqual(
                isCapture ? builder.PlaybackId : builder.CaptureId,
                info.id));

        if (existing is null)
        {
            existing = new DeviceBuilder(backend, name, isDefault);
            builders.Add(existing);
        }

        (int maxChannels, double defaultSampleRate) = GetFormatSummary(info);
        if (isCapture)
        {
            existing.CaptureId = info.id;
            existing.MaxInputChannels = maxChannels;
        }
        else
        {
            existing.PlaybackId = info.id;
            existing.MaxOutputChannels = maxChannels;
        }

        existing.DeviceType = existing.CaptureId.HasValue && existing.PlaybackId.HasValue
            ? AudioDeviceType.InputOutput
            : existing.CaptureId.HasValue
                ? AudioDeviceType.Input
                : AudioDeviceType.Output;

        if (existing.DefaultSampleRate <= 0 && defaultSampleRate > 0)
        {
            existing.DefaultSampleRate = defaultSampleRate;
        }
    }

    // DirectSound and WinMM populate miniaudio's fixed-size name buffer from the Win32 A APIs.
    // Decode those backends with the active Windows ANSI code page. Valid UTF-8 is not a reliable
    // discriminator here: for example, CP936 bytes C2 A9 are valid UTF-8 for U+00A9 but represent
    // the Chinese character "漏" in the ANSI device name.
    internal static unsafe string GetDeviceName(ma_device_info info, ma_backend backend)
    {
        // info is a local copy, so its fixed-size name buffer is already pinned.
        byte* name = info.name;
        int length = 0;
        while (length < DeviceNameBufferSize && name[length] != 0)
        {
            length++;
        }

        if (length == 0)
        {
            return string.Empty;
        }

        if (backend is ma_backend.dsound or ma_backend.winmm)
        {
            Encoding? ansi = AnsiEncoding.Value;
            return ansi is null ? info.GetName() : ansi.GetString(name, length);
        }

        try
        {
            return StrictUtf8.GetString(name, length);
        }
        catch (DecoderFallbackException)
        {
            Encoding? ansi = AnsiEncoding.Value;
            return ansi is null ? info.GetName() : ansi.GetString(name, length);
        }
    }

    private static Encoding? CreateAnsiEncoding()
    {
        try
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            // Code page 0 is the system's active code page, which is the one the Win32 ANSI
            // device APIs used to produce the raw name bytes.
            return Encoding.GetEncoding(0);
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"MiniAudio could not resolve the system ANSI code page: {exception.Message}");
            return null;
        }
    }

    private static unsafe ma_device_info GetDetailedDeviceInfo(
        ma_context_ptr context,
        ma_device_info basicInfo,
        bool isCapture)
    {
        ma_device_id_ptr id = new(true);
        if (id.pointer == IntPtr.Zero)
        {
            return basicInfo;
        }

        try
        {
            *id.Get() = basicInfo.id;
            ma_device_type deviceType = isCapture ? ma_device_type.capture : ma_device_type.playback;
            ma_result result = MiniAudioNative.ma_context_get_device_info(context, deviceType, id, out ma_device_info detailedInfo);
            return result == ma_result.success ? detailedInfo : basicInfo;
        }
        finally
        {
            id.Free();
        }
    }

    private static unsafe (int MaxChannels, double DefaultSampleRate) GetFormatSummary(ma_device_info info)
    {
        int count = (int)Math.Min(info.nativeDataFormatCount, 64u);
        int maxChannels = 0;
        double sampleRate = 0;
        for (int i = 0; i < count; i++)
        {
            ma_native_data_format format = info.nativeDataFormats[i];
            if (format.channels > 0)
            {
                maxChannels = Math.Max(maxChannels, (int)format.channels);
            }

            if (sampleRate <= 0 && format.sampleRate > 0)
            {
                sampleRate = format.sampleRate;
            }
        }

        return (maxChannels, sampleRate);
    }

    private static unsafe bool DeviceIdsEqual(ma_device_id? left, ma_device_id right)
    {
        if (!left.HasValue)
        {
            return false;
        }

        ma_device_id leftValue = left.Value;
        byte* leftBytes = (byte*)&leftValue;
        byte* rightBytes = (byte*)&right;
        for (int i = 0; i < sizeof(ma_device_id); i++)
        {
            if (leftBytes[i] != rightBytes[i])
            {
                return false;
            }
        }

        return true;
    }

    private sealed class DeviceBuilder(ma_backend backend, string name, bool isDefault)
    {
        public ma_backend Backend { get; } = backend;
        public string Name { get; } = name;
        public bool IsDefault { get; set; } = isDefault;
        public int MaxInputChannels { get; set; }
        public int MaxOutputChannels { get; set; }
        public double DefaultSampleRate { get; set; }
        public AudioDeviceType DeviceType { get; set; }
        public ma_device_id? CaptureId { get; set; }
        public ma_device_id? PlaybackId { get; set; }
    }
}

internal readonly record struct MiniAudioDeviceSelection(
    AudioDevice AudioDevice,
    ma_backend Backend,
    ma_device_id? CaptureId,
    ma_device_id? PlaybackId)
{
    internal bool SupportsCapture => CaptureId.HasValue;

    internal bool SupportsPlayback => PlaybackId.HasValue;
}
