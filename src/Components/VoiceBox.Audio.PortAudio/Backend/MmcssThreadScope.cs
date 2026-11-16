using System.Runtime.InteropServices;

namespace VoiceBox.Audio.PortAudio.Backend
{
    /// <summary>
    /// 把当前线程注册进 Windows MMCSS (多媒体类调度服务) 的任务类别, 让系统调度器
    /// 在高 CPU 负载下 (典型: 本地 TTS/LLM 推理吃满核) 仍优先调度音频处理线程。
    /// <para>
    /// 用法: <c>using var mmcss = MmcssThreadScope.TryEnter("Pro Audio");</c> —
    /// 注册失败 (非 Windows / avrt.dll 缺失 / 注册表无该任务类别) 时返回 null,
    /// 调用方无需分支处理, 行为静默降级为普通线程。Dispose 在同一线程上反注册。
    /// </para>
    /// </summary>
    internal sealed class MmcssThreadScope : IDisposable
    {
        private nint _handle;

        private MmcssThreadScope(nint handle) => _handle = handle;

        /// <summary>
        /// 尝试把当前线程注册进 MMCSS 任务类别 (如 "Pro Audio" / "Audio")。
        /// 失败返回 null, 不抛异常。
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
}
