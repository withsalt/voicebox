using System.Runtime.InteropServices;

namespace VoiceBox.Audio.PortAudio.Backend;

internal sealed class NativeStderrScope : IDisposable
{
    private const int StandardError = 2;
    private const int WriteOnly = 1;
    private const int Interrupted = 4;
    private static readonly NativeStderrScope NoOp = new(-1);
    private readonly int _savedStderr;

    private NativeStderrScope(int savedStderr) => _savedStderr = savedStderr;

    public static NativeStderrScope TryCreate()
    {
        if (!OperatingSystem.IsLinux())
        {
            return NoOp;
        }

        int savedStderr = -1;
        int nullDevice = -1;
        bool redirected = false;
        bool scopeCreated = false;

        try
        {
            savedStderr = Dup(StandardError);
            if (savedStderr < 0)
            {
                return NoOp;
            }

            nullDevice = Open("/dev/null", WriteOnly);
            if (nullDevice < 0 || Dup2(nullDevice, StandardError) < 0)
            {
                return NoOp;
            }

            redirected = true;
            Close(nullDevice);
            nullDevice = -1;

            var scope = new NativeStderrScope(savedStderr);
            savedStderr = -1;
            scopeCreated = true;
            return scope;
        }
        catch (Exception)
        {
            return NoOp;
        }
        finally
        {
            if (!scopeCreated)
            {
                if (redirected && savedStderr >= 0)
                {
                    RestoreStderr(savedStderr);
                }

                if (nullDevice >= 0)
                {
                    TryClose(nullDevice);
                }

                if (savedStderr >= 0)
                {
                    TryClose(savedStderr);
                }
            }
        }
    }

    public void Dispose()
    {
        if (_savedStderr < 0)
        {
            return;
        }

        if (RestoreStderr(_savedStderr))
        {
            TryClose(_savedStderr);
        }
    }

    private static bool RestoreStderr(int savedStderr)
    {
        while (true)
        {
            try
            {
                if (Dup2(savedStderr, StandardError) >= 0)
                {
                    return true;
                }

                if (Marshal.GetLastPInvokeError() != Interrupted)
                {
                    return false;
                }
            }
            catch (Exception ex) when (IsInteropUnavailable(ex))
            {
                return false;
            }
        }
    }

    private static void TryClose(int fileDescriptor)
    {
        try
        {
            Close(fileDescriptor);
        }
        catch (Exception ex) when (IsInteropUnavailable(ex))
        {
        }
    }

    private static bool IsInteropUnavailable(Exception exception) =>
        exception is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException;

    [DllImport("libc.so.6", EntryPoint = "dup", SetLastError = true)]
    private static extern int Dup(int fileDescriptor);

    [DllImport("libc.so.6", EntryPoint = "dup2", SetLastError = true)]
    private static extern int Dup2(int sourceFileDescriptor, int targetFileDescriptor);

    [DllImport("libc.so.6", EntryPoint = "open", SetLastError = true)]
    private static extern int Open(string path, int flags);

    [DllImport("libc.so.6", EntryPoint = "close", SetLastError = true)]
    private static extern int Close(int fileDescriptor);
}
