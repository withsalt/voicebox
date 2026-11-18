using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace VoiceBox.DNR.GRENet.Internal;

internal static class CudaDeviceMemory
{
    private const string DriverLibrary = "VoiceBox.CudaDriver";
    private const int Success = 0;

    static CudaDeviceMemory() => NativeLibrary.SetDllImportResolver(typeof(CudaDeviceMemory).Assembly, ResolveDriver);

    private static nint ResolveDriver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != DriverLibrary)
        {
            return 0;
        }

        if (OperatingSystem.IsLinux())
        {
            return NativeLibrary.Load("libcuda.so.1", assembly, searchPath);
        }

        if (OperatingSystem.IsWindows())
        {
            return NativeLibrary.Load("nvcuda.dll", assembly, DllImportSearchPath.System32);
        }

        throw new PlatformNotSupportedException("The GRE-Net CUDA runner requires Windows or Linux with an NVIDIA driver.");
    }

    public static nint GetAllocationContext(ulong pointer)
    {
        const int contextAttribute = 1;
        Check(CuPointerGetAttribute(out nint context, contextAttribute, pointer), "cuPointerGetAttribute");
        return context;
    }

    public readonly ref struct ContextScope
    {
        public ContextScope(nint context) => Check(CuCtxPushCurrent(context), "cuCtxPushCurrent");

        public void Dispose() => Check(CuCtxPopCurrent(out _), "cuCtxPopCurrent");
    }

    public static void FinishClearing() => Check(CuStreamSynchronize(0), "cuStreamSynchronize");

    public static unsafe ulong GetDevicePointer(Microsoft.ML.OnnxRuntime.OrtValue value)
    {
        Span<float> data = value.GetTensorMutableDataAsSpan<float>();
        ref float first = ref MemoryMarshal.GetReference(data);
        return (ulong)Unsafe.AsPointer(ref first);
    }

    public static unsafe void CopyHostToDevice(ulong destination, ReadOnlySpan<float> source)
    {
        if (source.IsEmpty)
        {
            return;
        }

        fixed (float* sourcePointer = source)
        {
            Check(CuMemcpyHtoD(destination, (nint)sourcePointer, (nuint)(source.Length * sizeof(float))), "cuMemcpyHtoD");
        }
    }

    public static unsafe void CopyDeviceToHost(Span<float> destination, ulong source)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        fixed (float* destinationPointer = destination)
        {
            Check(CuMemcpyDtoH((nint)destinationPointer, source, (nuint)(destination.Length * sizeof(float))), "cuMemcpyDtoH");
        }
    }

    public static void Clear(ulong destination, long byteCount)
    {
        Check(CuMemsetD8(destination, 0, (nuint)byteCount), "cuMemsetD8");
    }

    // Both allocations come from ORT rather than cuMemAllocPitch. Use the driver copy that accepts arbitrary pitches.
    [StructLayout(LayoutKind.Sequential)]
    internal struct RingCopy
    {
        private nuint _sourceX, _sourceY;
        private int _sourceType;
        private nint _sourceHost;
        private ulong _sourceDevice;
        private nint _sourceArray;
        private nuint _sourcePitch;
        private nuint _destinationX, _destinationY;
        private int _destinationType;
        private nint _destinationHost;
        private ulong _destinationDevice;
        private nint _destinationArray;
        private nuint _destinationPitch;
        private nuint _width, _height;

        public RingCopy(ulong frame, ulong ring, int rows, int slots, int channels)
        {
            const int deviceMemory = 2;
            _sourceType = _destinationType = deviceMemory;
            _sourceDevice = frame;
            _destinationDevice = ring;
            _sourcePitch = _width = checked((nuint)channels * sizeof(float));
            _destinationPitch = checked((nuint)slots * _width);
            _height = (nuint)rows;
        }

        public void Append()
        {
            Check(CuMemcpy2DUnaligned(in this), "cuMemcpy2DUnaligned");
            _destinationX = (_destinationX + _width) % _destinationPitch;
        }

        public void Reset() => _destinationX = 0;
    }

    [DllImport(DriverLibrary, EntryPoint = "cuMemcpy2DUnaligned_v2")]
    private static extern int CuMemcpy2DUnaligned(in RingCopy copy);

    private static void Check(int result, string operation)
    {
        if (result != Success)
        {
            throw new InvalidOperationException($"CUDA {operation} failed with error code {result}.");
        }
    }

    [DllImport(DriverLibrary, EntryPoint = "cuMemcpyHtoD_v2")]
    private static extern int CuMemcpyHtoD(ulong destination, nint source, nuint byteCount);

    [DllImport(DriverLibrary, EntryPoint = "cuMemcpyDtoH_v2")]
    private static extern int CuMemcpyDtoH(nint destination, ulong source, nuint byteCount);

    [DllImport(DriverLibrary, EntryPoint = "cuMemsetD8_v2")]
    private static extern int CuMemsetD8(ulong destination, byte value, nuint byteCount);

    [DllImport(DriverLibrary, EntryPoint = "cuPointerGetAttribute")]
    private static extern int CuPointerGetAttribute(out nint value, int attribute, ulong pointer);

    [DllImport(DriverLibrary, EntryPoint = "cuCtxPushCurrent_v2")]
    private static extern int CuCtxPushCurrent(nint context);

    [DllImport(DriverLibrary, EntryPoint = "cuCtxPopCurrent_v2")]
    private static extern int CuCtxPopCurrent(out nint context);

    [DllImport(DriverLibrary, EntryPoint = "cuStreamSynchronize")]
    private static extern int CuStreamSynchronize(nint stream);
}
