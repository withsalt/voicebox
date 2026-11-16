using System.Runtime.InteropServices;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 当 ExecutionProvider=Cuda 时, 把外部下载的 onnxruntime CUDA dll 目录顶到 PATH 最前面,
    /// 并劫持 onnxruntime.dll 的加载到该目录, 避开 NuGet 内置的 CUDA12/DirectML dll。
    /// 行为与服务端运行时加载逻辑保持一致。
    /// </summary>
    internal static class OnnxRuntimeNativeLoader
    {
        public static void Configure(string? nativeDir, OnnxExecutionProvider provider)
        {
            if (provider != OnnxExecutionProvider.Cuda) return;
            if (!OperatingSystem.IsWindows()) return;
            if (string.IsNullOrWhiteSpace(nativeDir)) return;
            if (!Directory.Exists(nativeDir))
            {
                Console.WriteLine($"[WARN] onnxruntime native dir not found: {nativeDir}");
                return;
            }

            string current = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            if (!current.Split(';').Any(p => string.Equals(p.TrimEnd('\\'), nativeDir.TrimEnd('\\'), StringComparison.OrdinalIgnoreCase)))
            {
                Environment.SetEnvironmentVariable("PATH", nativeDir + ";" + current);
            }

            NativeLibrary.SetDllImportResolver(
                typeof(Microsoft.ML.OnnxRuntime.OrtEnv).Assembly,
                (name, _, _) =>
                {
                    if (string.Equals(name, "onnxruntime", StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(name, "onnxruntime.dll", StringComparison.OrdinalIgnoreCase))
                    {
                        string full = Path.Combine(nativeDir, "onnxruntime.dll");
                        if (NativeLibrary.TryLoad(full, out var handle))
                            return handle;
                    }
                    return IntPtr.Zero;
                });
        }
    }
}
