// License:     APL 2.0
// Author:      Benjamin N. Summerton <https://16bpp.net>

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using SharpASR.AudioService.Backend;
using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Audio.PortAudio.Backend.Enums;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Backend
{
    internal static class PortAudioBackend
    {
        #region Constants
        /// <summary>
        /// A special PaDeviceIndex value indicating that no device is available,
        /// or should be used.
        ///
        /// @see PaDeviceIndex
        /// </summary>
        public const int NoDevice = -1;

        /// <summary>
        /// Can be passed as the framesPerBuffer parameter to Pa_OpenStream()
        /// or Pa_OpenDefaultStream() to indicate that the stream callback will
        /// accept buffers of any size.
        /// </summary>
        public const uint FramesPerBufferUnspecified = 0;
        #endregion // Constants

        /// <summary>
        /// Retrieve the release number of the currently running PortAudio build.
        /// For example, for version "19.5.1" this will return 0x00130501.
        ///
        /// @see paMakeVersionNumber
        /// </summary>
        /// <value></value>
        public static int Version
        {
            get => PortAudioNative.Pa_GetVersion();
        }

        /// <summary>
        /// Retrieve version information for the currently running PortAudio build.
        /// @return A pointer to an immutable PaVersionInfo structure.
        ///
        /// @note This function can be called at any time. It does not require PortAudio
        /// to be initialized. The structure pointed to is statically allocated. Do not
        /// attempt to free it or modify it.
        ///
        /// @see PaVersionInfo, paMakeVersionNumber
        /// @version Available as of 19.5.0.
        /// </summary>
        public static VersionInfo VersionInfo
        {
            get => Marshal.PtrToStructure<VersionInfo>(PortAudioNative.Pa_GetVersionInfo());
        }

        /// <summary>
        /// Translate the supplied PortAudio error code into a human readable
        /// message.
        /// </summary>
        public static string? GetErrorText(ErrorCode errorCode) =>
            Marshal.PtrToStringAnsi(PortAudioNative.Pa_GetErrorText(errorCode));

        /// <summary>
        /// This is a function that's not found in the original PortAudio library.  Because of how the native libraries are
        /// packaged, this function must be called before anything else with the package.  It loads the native shared library
        /// (from an embedded resource) and makes it accessible.
        /// </summary>
        public static void LoadNativeLibrary()
        {
            /*
                Commented out in development branch so `/third_party/lib` DLLs can be used instead

            // Extract the the native library that has been embedded and load it up
            ResourceAccessor accessor = new ResourceAccessor(Assembly.GetExecutingAssembly());
            LibraryManager libManager = new LibraryManager(
                Assembly.GetExecutingAssembly(),
                new LibraryItem(Platform.Linux,   Bitness.x64, new LibraryFile("libportaudio.so",    accessor.Binary("libportaudio.so"))),
                new LibraryItem(Platform.MacOs,   Bitness.x64, new LibraryFile("libportaudio.dylib", accessor.Binary("libportaudio.dylib"))),
                new LibraryItem(Platform.Windows, Bitness.x64, new LibraryFile("portaudio.dll",      accessor.Binary("portaudio.dll")))
            );
            libManager.LoadNativeLibrary();
            */
        }

        /// <summary>
        /// Library initialization function - call this before using PortAudio.
        /// This function initializes internal data structures and prepares underlying
        /// host APIs for use.  With the exception of Pa_GetVersion(), Pa_GetVersionText(),
        /// and Pa_GetErrorText(), this function MUST be called before using any other
        /// PortAudio API functions.
        ///
        /// If Pa_Initialize() is called multiple times, each successful
        /// call must be matched with a corresponding call to Pa_Terminate().
        /// Pairs of calls to Pa_Initialize()/Pa_Terminate() may overlap, and are not
        /// required to be fully nested.
        ///
        /// Note that if Pa_Initialize() returns an error code, Pa_Terminate() should
        /// NOT be called.
        ///
        /// @return paNoError if successful, otherwise an error code indicating the cause
        /// of failure.
        ///
        /// @see Pa_Terminate
        /// </summary>
        public static void Initialize()
        {
            // PortAudio probes optional host APIs here; their native diagnostics are not PortAudio errors.
            using (var stderrScope = NativeStderrScope.TryCreate())
            {
                ErrorCode ec = PortAudioNative.Pa_Initialize();
                if (ec != ErrorCode.NoError)
                    throw new PortAudioException(ec, "Error initializing PortAudio");
            }
        }

        /// <summary>
        /// Library termination function - call this when finished using PortAudio.
        /// This function deallocates all resources allocated by PortAudio since it was
        /// initialized by a call to Pa_Initialize(). In cases where Pa_Initialise() has
        /// been called multiple times, each call must be matched with a corresponding call
        /// to Pa_Terminate(). The final matching call to Pa_Terminate() will automatically
        /// close any PortAudio streams that are still open.
        ///
        /// Pa_Terminate() MUST be called before exiting a program which uses PortAudio.
        /// Failure to do so may result in serious resource leaks, such as audio devices
        /// not being available until the next reboot.
        ///
        /// @return paNoError if successful, otherwise an error code indicating the cause
        /// of failure.
        ///
        /// @see Pa_Initialize
        /// </summary>
        public static void Terminate()
        {
            ErrorCode ec = PortAudioNative.Pa_Terminate();
            if (ec != ErrorCode.NoError)
                throw new PortAudioException(ec, "Error terminating PortAudio");
        }

        /// <summary>
        /// Retrieve the index of the default output device. The result can be
        /// used in the outputDevice parameter to Pa_OpenStream().
        ///
        /// @return The default output device index for the default host API, or paNoDevice
        /// if no default output device is available or an error was encountered.
        ///
        /// @note
        /// On the PC, the user can specify a default device by
        /// setting an environment variable. For example, to use device #1.
        /// <pre>
        /// set PA_RECOMMENDED_OUTPUT_DEVICE=1
        /// </pre>
        /// The user should first determine the available device ids by using
        /// the supplied application "pa_devs".
        /// </summary>
        public static int DefaultOutputDevice
        {
            get => PortAudioNative.Pa_GetDefaultOutputDevice();
        }

        /// <summary>
        /// Retrieve the index of the default input device. The result can be
        /// used in the inputDevice parameter to Pa_OpenStream().
        ///
        /// @return The default input device index for the default host API, or paNoDevice
        /// if no default input device is available or an error was encountered.
        /// </summary>
        public static int DefaultInputDevice
        {
            get => PortAudioNative.Pa_GetDefaultInputDevice();
        }

        /// <summary>
        /// Retrieve a pointer to a PaDeviceInfo structure containing information
        /// about the specified device.
        /// @return A pointer to an immutable PaDeviceInfo structure. If the device
        /// parameter is out of range the function returns NULL.
        ///
        /// @param device A valid device index in the range 0 to (Pa_GetDeviceCount()-1)
        ///
        /// @note PortAudio manages the memory referenced by the returned pointer,
        /// the client must not manipulate or free the memory. The pointer is only
        /// guaranteed to be valid between calls to Pa_Initialize() and Pa_Terminate().
        ///
        /// @see PaDeviceInfo, PaDeviceIndex
        /// </summary>
        public static DeviceInfo? GetDeviceInfo(int device)
        {
            nint ptr = PortAudioNative.Pa_GetDeviceInfo(device);
            if (ptr == nint.Zero)
            {
                Debug.WriteLine($"PortAudio failed to get device info for index {device}.");
                return null;
            }

            var info = Marshal.PtrToStructure<DeviceInfoData>(ptr);
            DeviceInfo ans = new DeviceInfo();

            ans.structVersion = info.structVersion;
            ans.name = GetDeviceName(info.name);
            ans.hostApi = info.hostApi;
            ans.maxInputChannels = info.maxInputChannels;
            ans.maxOutputChannels = info.maxOutputChannels;
            ans.defaultLowInputLatency = info.defaultLowInputLatency;
            ans.defaultLowOutputLatency = info.defaultLowOutputLatency;
            ans.defaultHighInputLatency = info.defaultHighInputLatency;
            ans.defaultHighOutputLatency = info.defaultHighOutputLatency;
            ans.defaultSampleRate = info.defaultSampleRate;

            return ans;
        }

        private static string GetDeviceName(nint name)
        {
            string s = "";
            if (name != nint.Zero)
            {
                int length = 0;
                unsafe
                {
                    byte* b = (byte*)name;
                    if (b != null)
                    {
                        while (*b != 0)
                        {
                            ++b;
                            length += 1;
                        }
                    }
                }

                if (length > 0)
                {
                    byte[] stringBuffer = new byte[length];
                    Marshal.Copy(name, stringBuffer, 0, length);
                    s = Encoding.UTF8.GetString(stringBuffer);
                }
            }
            return s;
        }

        /// <summary>
        /// Retrieve the number of available devices. The number of available devices
        /// may be zero.
        ///
        /// @return A non-negative value indicating the number of available devices
        /// or, a PaErrorCode (which are always negative) if PortAudio is not initialized
        /// or an error is encountered.
        /// </summary>
        public static int DeviceCount
        {
            get => PortAudioNative.Pa_GetDeviceCount();
        }

        /// <summary>
        /// ��ȡָ�� Host API ����ϸ��Ϣ��
        /// </summary>
        /// <param name="hostApi">Ҫ��ѯ�� Host API ������</param>
        /// <returns>һ������ API ��ϸ��Ϣ�� HostApiInfo �ṹ�塣</returns>
        /// <exception cref="InvalidOperationException">�� PortAudio �ײ����ʧ��ʱ�׳���</exception>
        public static HostApiInfo? GetHostApiInfo(int hostApi)
        {
            nint apiInfoPtr = PortAudioNative.Pa_GetHostApiInfo(hostApi);
            if (apiInfoPtr == nint.Zero)
            {
                Debug.WriteLine($"PortAudio failed to get Host API info for index {hostApi}.");
                return null; // Return null if the pointer is invalid
            }

            return Marshal.PtrToStructure<HostApiInfo>(apiInfoPtr);
        }

        /// <summary>
        /// 把 host API 类型映射为当前运行时的 host API 索引; 该类型未编译进库或不可用时返回负值。
        /// </summary>
        public static int GetHostApiIndex(HostApiType type) =>
            PortAudioNative.Pa_HostApiTypeIdToHostApiIndex((int)type);

        /// <summary>
        /// 首选 host API: Windows 上为 WASAPI (设备名完整不截断、共享模式延迟远低于 MME),
        /// 其他平台跟随系统默认。返回 (-1, null) 表示无首选项, 调用方应回退到 PortAudio 全局默认。
        /// </summary>
        public static (int Index, HostApiInfo? Info) GetPreferredHostApi()
        {
            if (OperatingSystem.IsWindows())
            {
                int wasapi = GetHostApiIndex(HostApiType.WASAPI);
                if (wasapi >= 0)
                {
                    HostApiInfo? info = GetHostApiInfo(wasapi);
                    if (info != null)
                    {
                        return (wasapi, info);
                    }
                }
            }
            return (-1, null);
        }

        /// <summary>
        /// 默认输入设备 (全局设备索引)。Windows 上取 WASAPI host API 的默认输入设备,
        /// WASAPI 不可用或无输入设备时回退到 <see cref="DefaultInputDevice"/> (默认 host API, 即 MME)。
        /// </summary>
        public static int PreferredDefaultInputDevice
        {
            get
            {
                (_, HostApiInfo? info) = GetPreferredHostApi();
                if (info is { defaultInputDevice: >= 0 })
                {
                    return info.Value.defaultInputDevice;
                }
                return DefaultInputDevice;
            }
        }

        /// <summary>
        /// 默认输出设备 (全局设备索引)。Windows 上取 WASAPI host API 的默认输出设备,
        /// WASAPI 不可用或无输出设备时回退到 <see cref="DefaultOutputDevice"/>。
        /// </summary>
        public static int PreferredDefaultOutputDevice
        {
            get
            {
                (_, HostApiInfo? info) = GetPreferredHostApi();
                if (info is { defaultOutputDevice: >= 0 })
                {
                    return info.Value.defaultOutputDevice;
                }
                return DefaultOutputDevice;
            }
        }

        /// <summary>判断设备是否属于 WASAPI host API。</summary>
        public static bool IsWasapiDevice(int deviceIndex)
        {
            if (deviceIndex < 0)
            {
                return false;
            }
            DeviceInfo? deviceInfo = GetDeviceInfo(deviceIndex);
            if (deviceInfo == null)
            {
                return false;
            }
            HostApiInfo? hostApiInfo = GetHostApiInfo(deviceInfo.Value.hostApi);
            return hostApiInfo?.type == HostApiType.WASAPI;
        }

        /// <summary>
        /// 为 WASAPI 设备分配一份非托管 <see cref="PaWasapiStreamInfo"/>, 经
        /// StreamParameters.hostApiSpecificStreamInfo 传给 Pa_OpenStream。
        /// 共享模式带 AutoConvert (允许 16kHz 等与系统混音格式不一致的流由系统重采样),
        /// 独占模式带 Exclusive (AutoConvert 仅共享模式有效, 二者不叠加);
        /// 两种模式均请求 MMCSS "Pro Audio" 线程优先级。
        /// <para>
        /// Pa_OpenStream 在打开时把该结构 memcpy 进流自身状态, 返回后调用方必须
        /// <see cref="Marshal.FreeHGlobal"/> 释放。设备不属于 WASAPI 时返回 <see cref="nint.Zero"/>。
        /// </para>
        /// </summary>
        public static nint AllocWasapiStreamInfo(int deviceIndex, bool exclusiveMode)
        {
            if (!IsWasapiDevice(deviceIndex))
            {
                return nint.Zero;
            }

            // ThreadPriority + ProAudio: 让 PortAudio 的 WASAPI 处理线程注册到 MMCSS "Pro Audio"
            // 任务类别 (不设时默认 "Audio")。回调内跑着 HPF/AEC3 等采集处理, 系统高负载时
            // (典型: TTS 推理吃满 CPU) 更高的 MMCSS 类别可显著降低回调抖动, 保住 AEC 收敛。
            var info = new PaWasapiStreamInfo
            {
                size = (uint)Marshal.SizeOf<PaWasapiStreamInfo>(),
                hostApiType = (int)HostApiType.WASAPI,
                version = 1,
                flags = (uint)((exclusiveMode ? PaWasapiFlags.Exclusive : PaWasapiFlags.AutoConvert)
                    | PaWasapiFlags.ThreadPriority),
                threadPriority = (int)PaWasapiThreadPriority.ProAudio,
            };

            nint ptr = Marshal.AllocHGlobal(Marshal.SizeOf<PaWasapiStreamInfo>());
            Marshal.StructureToPtr(info, ptr, false);
            return ptr;
        }

        /// <summary>释放 <see cref="AllocWasapiStreamInfo"/> 分配的非托管内存; 传入 <see cref="nint.Zero"/> 时为 no-op。</summary>
        public static void FreeWasapiStreamInfo(nint ptr)
        {
            if (ptr != nint.Zero)
            {
                Marshal.FreeHGlobal(ptr);
            }
        }
    }
}
