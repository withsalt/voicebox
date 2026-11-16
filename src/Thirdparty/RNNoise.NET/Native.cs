using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace RNNoise.NET
{
    public static partial class Native
    {
        public const string LIBRARY_NAME = "rnnoise";

        public const int FRAME_SIZE = 480;

        public const float SIGNAL_SCALE = short.MaxValue;
        public const float SIGNAL_SCALE_INV = 1f / short.MaxValue;

        static Native()
        {
            NativeLibraryLoader.EnsureRegistered();
        }

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int rnnoise_get_size();

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int rnnoise_init(IntPtr state, IntPtr model);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr rnnoise_create(IntPtr model);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void rnnoise_destroy(IntPtr state);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static unsafe partial float rnnoise_process_frame(IntPtr state, float* dataOut, float* dataIn);

        [LibraryImport(LIBRARY_NAME, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr rnnoise_system_open_file(string fileName, string mode);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial int rnnoise_system_close_file(IntPtr file);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial IntPtr rnnoise_model_from_file(IntPtr file);

        [LibraryImport(LIBRARY_NAME)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        public static partial void rnnoise_model_free(IntPtr model);
    }
}
