namespace VoiceBox.Core.Services.AECService
{
    /// <summary>
    /// 声学回声消除服务。
    /// 实现应在全双工音频回调中使用与采集时间对齐的播放(render)参考信号，
    /// 原地改写麦克风采集(capture)信号。
    /// <para>
    /// 一个统一的采集侧处理器(如 WebRTC APM)也可以此身份注册，从而无需改动全双工服务的接线。
    /// </para>
    /// <para><see cref="Process"/> 须在单一(实时)线程调用，不应与 <see cref="Reset"/> 并发。</para>
    /// </summary>
    public interface IAcousticEchoCancellationService : IDisposable
    {
        /// <summary>
        /// 获取处理器为保持流式等长输出而引入的固定延迟。
        /// 返回值必须大于或等于 0，单位为处理器配置格式下的交织样本数，
        /// 不包含播放参考信号的声学路径对齐延迟。
        /// </summary>
        int DelaySamples { get; }

        /// <summary>
        /// 处理一帧音频。
        /// </summary>
        /// <param name="capture">近端(麦克风)采集信号；实现应原地改写为处理后的结果。</param>
        /// <param name="render">与 <paramref name="capture"/> 时间对齐的远端(播放)参考信号，只读。</param>
        /// <param name="channels">声道数（交织排列）。</param>
        /// <param name="sampleRate">采样率 (Hz)。</param>
        void Process(Span<float> capture, ReadOnlySpan<float> render, uint channels, uint sampleRate);

        /// <summary>
        /// 重置内部状态（缓冲、自适应滤波器等）。应在音频流停止/重启时调用，不应与 <see cref="Process"/> 并发。
        /// </summary>
        void Reset();
    }
}
