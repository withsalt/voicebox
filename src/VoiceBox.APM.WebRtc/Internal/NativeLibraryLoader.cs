using System.Reflection;
using System.Runtime.InteropServices;

namespace VoiceBox.APM.WebRtc.Internal
{
    /// <summary>
    /// 为原生 <c>webrtc-apm</c> 库配置加载路径。参考 <c>WeTextProcessing.NET</c> 的做法：
    /// 原生库以 <c>runtimes/&lt;rid&gt;/native/</c> 结构随生成目录拷贝，本类注册一个
    /// <see cref="NativeLibrary.SetDllImportResolver"/> 解析器，按当前 RID 在生成目录下
    /// 定位并加载对应原生库。
    /// <para>
    /// 尽力而为：若在 <c>runtimes/&lt;rid&gt;/native/</c> 找到则按绝对路径加载；否则回退到 .NET
    /// 默认原生库搜索(NuGet runtime pack、系统 PATH 等)，不抛异常。
    /// </para>
    /// </summary>
    public static class NativeLibraryLoader
    {
        private const string LibraryName = "webrtc-apm";

        private static string? s_nativeLibraryPath;
        private static bool s_resolverRegistered;
        private static readonly object s_lock = new();

        /// <summary>
        /// 确保解析器已注册。首次调用时定位 <c>runtimes/&lt;rid&gt;/native/</c> 下的原生库并记录其绝对路径，
        /// 之后 <c>[LibraryImport("webrtc-apm")]</c> 的解析都会命中该路径。多次调用幂等、线程安全。
        /// 应在首次 P/Invoke 之前调用。
        /// </summary>
        public static void EnsureRegistered()
        {
            lock (s_lock)
            {
                s_nativeLibraryPath ??= TryResolveBundledLibraryPath();
                EnsureResolverRegisteredCore();
            }
        }

        private static void EnsureResolverRegisteredCore()
        {
            if (s_resolverRegistered) return;
            NativeLibrary.SetDllImportResolver(typeof(ApmInterop).Assembly, Resolve);
            s_resolverRegistered = true;
        }

        private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != LibraryName)
            {
                return nint.Zero;
            }

            string? path;
            lock (s_lock)
            {
                path = s_nativeLibraryPath;
            }

            if (path is not null && NativeLibrary.TryLoad(path, out nint handle))
            {
                return handle;
            }

            // 回退到运行时默认解析。
            return nint.Zero;
        }

        private static string? TryResolveBundledLibraryPath()
        {
            try
            {
                return ResolveBundledLibraryPath();
            }
            catch (FileNotFoundException)
            {
                return null;
            }
            catch (PlatformNotSupportedException)
            {
                return null;
            }
        }

        private static string ResolveBundledLibraryPath()
        {
            string rid = GetRuntimeIdentifierPath();
            string fileName = GetNativeLibraryFileName();
            string relative = Path.Combine("runtimes", rid, "native", fileName);
            return RuntimeAssetLocator.FindRequired(relative, $"Native library '{fileName}'");
        }

        private static string GetRuntimeIdentifierPath()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.X64 => "win-x64",
                    Architecture.X86 => "win-x86",
                    _ => throw new PlatformNotSupportedException(
                        $"Unsupported Windows processor architecture: {RuntimeInformation.ProcessArchitecture}"),
                };
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.X64 => "linux-x64",
                    Architecture.Arm64 => "linux-arm64",
                    Architecture.Arm => "linux-arm",
                    _ => throw new PlatformNotSupportedException(
                        $"Unsupported Linux processor architecture: {RuntimeInformation.ProcessArchitecture}"),
                };
            }

            throw new PlatformNotSupportedException(
                $"Unsupported operating system: {RuntimeInformation.OSDescription}");
        }

        private static string GetNativeLibraryFileName()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return $"{LibraryName}.dll";
            }
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return $"lib{LibraryName}.so";
            }

            throw new PlatformNotSupportedException(
                $"webrtc-apm native library is not supported on the current platform: {RuntimeInformation.OSDescription}");
        }
    }
}
