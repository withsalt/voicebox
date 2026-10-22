#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace WeTextProcessing.NET.Interop;

/// <summary>
/// Provides a mechanism to configure the native library search path for the WeTextProcessing native library.
/// </summary>
public static class NativeLibraryLoader
{
    private const string LibraryName = "wetext_processor_c";

    private static string? s_nativeLibraryPath;
    private static bool s_resolverRegistered;
    private static readonly object s_lock = new();

    /// <summary>
    /// Ensures the import resolver is registered. Best-effort: if the bundled
    /// library is found under <c>runtimes/{rid}/native/</c>, its absolute path is
    /// recorded for direct loading; otherwise the resolver simply falls back to
    /// the default .NET native-library search (NuGet runtime pack, system PATH,
    /// etc.) without throwing.
    /// </summary>
    public static void EnsureRegistered()
    {
        lock (s_lock)
        {
            if (s_nativeLibraryPath is null)
            {
                s_nativeLibraryPath = TryResolveBundledLibraryPath();
            }
            EnsureResolverRegisteredCore();
        }
    }

    private static void EnsureResolverRegisteredCore()
    {
        if (s_resolverRegistered) return;
        NativeLibrary.SetDllImportResolver(typeof(NativeMethods).Assembly, Resolve);
        s_resolverRegistered = true;
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
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

        if (path is not null && NativeLibrary.TryLoad(path, out var handle))
        {
            return handle;
        }

        // Fall back to the runtime's default resolution.
        return IntPtr.Zero;
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

        foreach (var baseDir in EnumerateProbeBaseDirectories())
        {
            var candidate = Path.Combine(baseDir, relative);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException(
            $"The native library '{fileName}' was not found under 'runtimes/{rid}/native/' " +
            "in any probed directory (AppContext.BaseDirectory, ./bin, assembly location).");
    }

    private static IEnumerable<string> EnumerateProbeBaseDirectories()
    {
        var baseDir = AppContext.BaseDirectory;
        if (!string.IsNullOrEmpty(baseDir))
        {
            yield return baseDir;
            // Web hosts (e.g. IIS, ASP.NET Core out-of-process) may place native
            // assets under a "bin" subdirectory of the content root.
            yield return Path.Combine(baseDir, "bin");
        }

        var assemblyLocation = Path.GetDirectoryName(typeof(NativeLibraryLoader).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyLocation) &&
            !string.Equals(assemblyLocation, baseDir, StringComparison.OrdinalIgnoreCase))
        {
            yield return assemblyLocation;
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
            $"WeTextProcessing native library is not supported on the current platform: {RuntimeInformation.OSDescription}");
    }
}
