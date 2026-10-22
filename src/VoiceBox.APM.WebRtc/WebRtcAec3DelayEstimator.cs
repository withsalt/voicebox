using VoiceBox.APM.WebRtc.Internal;
using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.APM.WebRtc
{
    /// <summary>
    /// WebRTC AEC3 回声路径延迟估计器(<c>webrtc::EchoPathDelayEstimator</c>)的托管封装。
    /// 这是 AEC3 实际使用的匹配滤波延迟估计器(降采样匹配滤波 + 多滤波器滞后聚合 + 时钟漂移检测)，
    /// 以独立、<b>时域</b>的形式暴露:送入时间对齐的 render(远端参考)/capture(近端) 样本，
    /// 取回<b>全采样级</b>的"render→capture"延迟(样本数)。
    /// <para>
    /// 不进行任何回声消除/降噪，仅做延迟估计。典型用于把延迟喂给另一套(如 DTLN)AEC 的参考延迟线。
    /// 当前仅支持 16kHz 单声道。<see cref="Process"/> 须单线程调用，不应与 <see cref="Reset"/>/
    /// <see cref="Dispose"/> 并发。
    /// </para>
    /// </summary>
    internal sealed class WebRtcAec3DelayEstimator : IAcousticPathDelayEstimator
    {
        /// <summary>当前实现支持的采样率。</summary>
        public const int SupportedSampleRate = 16000;

        /// <summary>当前实现支持的声道数。</summary>
        public const int SupportedChannels = 1;

        private nint _handle; // 原生句柄(0 表示已释放)
        private bool _disposed;

        static WebRtcAec3DelayEstimator()
        {
            // 注册原生库解析器并校验 ABI，须在任何 P/Invoke 之前。
            NativeLibraryLoader.EnsureRegistered();

            int native;
            try
            {
                native = ApmInterop.AbiVersion();
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException(
                    "Native library 'webrtc-apm' not found. Ensure the runtimes/<rid>/native binary is present " +
                    "(build it from native/webrtc-apm, see its README).", ex);
            }

            if (native != ApmInterop.ExpectedAbiVersion)
            {
                throw new InvalidOperationException(
                    $"webrtc-apm ABI mismatch: native={native}, expected={ApmInterop.ExpectedAbiVersion}. " +
                    "Rebuild the native shim from native/webrtc-apm.");
            }
        }

        /// <summary>创建一个延迟估计器实例。</summary>
        /// <param name="sampleRate">采样率，当前仅支持 16000。</param>
        /// <param name="channels">声道数，当前仅支持 1。</param>
        public WebRtcAec3DelayEstimator(int sampleRate = SupportedSampleRate, int channels = SupportedChannels)
        {
            if (sampleRate != SupportedSampleRate || channels != SupportedChannels)
            {
                throw new ArgumentException(
                    $"WebRtcAec3DelayEstimator supports {SupportedSampleRate} Hz mono only, got {sampleRate} Hz / {channels} ch.");
            }

            int rc = ApmInterop.DelayCreate(sampleRate, channels, out _handle);
            if (rc != ApmInterop.Ok || _handle == nint.Zero)
            {
                throw new InvalidOperationException(
                    $"apm_delay_create failed (rc={rc}). See native status codes in apm_shim.h.");
            }
        }

        /// <summary>
        /// 送入等长、时间对齐的 render/capture 样本。原生估计器仍在预热时返回
        /// <see langword="false"/>；原生处理失败时抛出异常，不会折叠为“无估计”。
        /// </summary>
        public bool TryEstimate(
            ReadOnlySpan<float> render,
            ReadOnlySpan<float> capture,
            out AcousticPathDelayEstimate estimate)
        {
            ObjectDisposedException.ThrowIf(_disposed || _handle == nint.Zero, this);
            if (render.Length != capture.Length)
            {
                throw new ArgumentException(
                    "Render and capture sample counts must match.",
                    nameof(capture));
            }

            estimate = default;
            if (render.IsEmpty)
                return false;

            unsafe
            {
                fixed (float* r = render)
                fixed (float* c = capture)
                {
                    int delay = 0;
                    int quality = 0;
                    int rc = ApmInterop.DelayProcess(
                        _handle,
                        r,
                        c,
                        render.Length,
                        &delay,
                        &quality);
                    if (rc == 0)
                        return false;
                    if (rc != 1)
                    {
                        throw new InvalidOperationException(
                            $"WebRTC AEC3 delay estimation failed with native status {rc}.");
                    }

                    estimate = new AcousticPathDelayEstimate(
                        Math.Max(0, delay),
                        quality != 0);
                    return true;
                }
            }
        }

        /// <summary>重置估计器内部状态(匹配滤波器/缓冲/累积)，保留配置。流停止/重启时调用。</summary>
        public void Reset()
        {
            if (_disposed || _handle == nint.Zero)
                return;
            ApmInterop.DelayReset(_handle);
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_handle != nint.Zero)
            {
                ApmInterop.DelayDestroy(_handle);
                _handle = nint.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~WebRtcAec3DelayEstimator()
        {
            if (_handle != nint.Zero)
            {
                try
                {
                    ApmInterop.DelayDestroy(_handle);
                }
                catch
                {
                    // 进程退出时原生库可能已卸载,吞掉异常避免终结器线程崩溃
                }
                _handle = nint.Zero;
            }
        }
    }
}
