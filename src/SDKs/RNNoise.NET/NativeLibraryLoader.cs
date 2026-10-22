#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace RNNoise.NET
{
    internal static class NativeLibraryLoader
    {
        private const string LibraryName = Native.LIBRARY_NAME;

        private static readonly object s_lock = new();
        private static string? s_nativeLibraryPath;
        private static bool s_resolverRegistered;

        internal static string? ResolvedNativeLibraryPath
        {
            get
            {
                EnsureRegistered();
                lock (s_lock)
                    return s_nativeLibraryPath;
            }
        }

        public static void EnsureRegistered()
        {
            lock (s_lock)
            {
                if (s_nativeLibraryPath is null)
                {
                    s_nativeLibraryPath = TryResolveBundledLibraryPath();
                }

                if (!s_resolverRegistered)
                {
                    NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
                    s_resolverRegistered = true;
                }
            }
        }

        private static IntPtr Resolve(
            string libraryName,
            Assembly assembly,
            DllImportSearchPath? searchPath)
        {
            if (libraryName != LibraryName)
            {
                return IntPtr.Zero;
            }

            string? path;
            lock (s_lock)
            {
                path = s_nativeLibraryPath;
            }

            return path is not null && NativeLibrary.TryLoad(path, out var handle)
                ? handle
                : IntPtr.Zero;
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
            string relativePath = Path.Combine("runtimes", rid, "native", fileName);

            foreach (var baseDirectory in EnumerateProbeBaseDirectories())
            {
                string candidate = Path.Combine(baseDirectory, relativePath);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            throw new FileNotFoundException(
                $"The native library '{fileName}' was not found under 'runtimes/{rid}/native/'.");
        }

        [UnconditionalSuppressMessage(
            "SingleFile",
            "IL3000:Avoid accessing Assembly file path when publishing as a single file",
            Justification = "Assembly.Location is only an optional fallback; AppContext.BaseDirectory is probed first.")]
        private static IEnumerable<string> EnumerateProbeBaseDirectories()
        {
            string baseDirectory = AppContext.BaseDirectory;
            if (!string.IsNullOrEmpty(baseDirectory))
            {
                yield return baseDirectory;
                yield return Path.Combine(baseDirectory, "bin");
            }

            string? assemblyDirectory = Path.GetDirectoryName(typeof(Native).Assembly.Location);
            if (!string.IsNullOrEmpty(assemblyDirectory) &&
                !string.Equals(assemblyDirectory, baseDirectory, StringComparison.OrdinalIgnoreCase))
            {
                yield return assemblyDirectory;
            }
        }

        private static string GetRuntimeIdentifierPath()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return RuntimeInformation.ProcessArchitecture switch
                {
                    Architecture.X64 => "win-x64",
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
                $"RNNoise native library is not supported on the current platform: {RuntimeInformation.OSDescription}");
        }
    }
}
