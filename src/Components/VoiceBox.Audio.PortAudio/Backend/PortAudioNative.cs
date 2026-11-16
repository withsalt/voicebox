using System.Runtime.InteropServices;
using VoiceBox.Audio.PortAudio.Backend.Enums;

namespace SharpASR.AudioService.Backend
{
    internal static partial class PortAudioNative
    {
#if !IOS
        public const string PortAudioDLL = "portaudio";
#else
            public const string PortAudioDLL = "__Internal";
#endif

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Pa_GetVersion();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Pa_GetVersionInfo();           // Originally returns `const PaVersionInfo *`

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Pa_GetErrorText([MarshalAs(UnmanagedType.I4)] ErrorCode errorCode);     // Orignially returns `const char *`

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        public static extern ErrorCode Pa_Initialize();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I4)]
        public static extern ErrorCode Pa_Terminate();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Pa_GetDefaultOutputDevice();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Pa_GetDefaultInputDevice();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Pa_GetDeviceInfo(int device);   // Originally returns `const PaDeviceInfo *`

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Pa_GetDeviceCount();

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Pa_Sleep(System.Int32 msec);

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl, EntryPoint = "Pa_GetHostApiInfo")]
        public static extern IntPtr Pa_GetHostApiInfo(int hostApi);

        [DllImport(PortAudioDLL, CallingConvention = CallingConvention.Cdecl)]
        public static extern int Pa_HostApiTypeIdToHostApiIndex(int type);   // PaHostApiTypeId -> PaHostApiIndex, 不可用时返回负的错误码
    }
}
