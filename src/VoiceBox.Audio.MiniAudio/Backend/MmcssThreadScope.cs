using System.Runtime.InteropServices;

namespace VoiceBox.Audio.MiniAudio.Backend;

/// <summary>
/// Registers the current thread with a Windows MMCSS (Multimedia Class Scheduler Service) task
/// category so the scheduler keeps preferring it under high CPU load, which for this project means
/// local TTS/LLM inference saturating every core.
/// <para>
/// Usage: <c>using var mmcss = MmcssThreadScope.TryEnter("Pro Audio");</c> — returns null when
/// registration is not possible (not Windows, avrt.dll missing, task category absent), so callers
/// need no special case and silently degrade to a normal thread. Dispose deregisters on the same
/// thread.
/// </para>
/// </summary>
internal sealed class MmcssThreadScope : IDisposable
{
    private nint _handle;

    private MmcssThreadScope(nint handle) => _handle = handle;

    /// <summary>
    /// Tries to register the current thread with an MMCSS task category (for example
    /// "Pro Audio"). Returns null instead of throwing on failure.
    /// </summary>
    public static MmcssThreadScope? TryEnter(string taskName)
    {
        if (!OperatingSystem.IsWindows())
            return null;

        try
        {
            int taskIndex = 0;
            nint handle = AvSetMmThreadCharacteristicsW(taskName, ref taskIndex);
            return handle != nint.Zero ? new MmcssThreadScope(handle) : null;
        }
        catch (DllNotFoundException)
        {
            return null;
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }
    }

    public void Dispose()
    {
        nint handle = _handle;
        _handle = nint.Zero;
        if (handle != nint.Zero)
        {
            try
            {
                AvRevertMmThreadCharacteristics(handle);
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }
    }

    [DllImport("avrt.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern nint AvSetMmThreadCharacteristicsW(string taskName, ref int taskIndex);

    [DllImport("avrt.dll", ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool AvRevertMmThreadCharacteristics(nint handle);
}
