namespace VoiceBox.Core.Services.HPFService
{
    /// <summary>
    /// 高通滤波(High-Pass Filter)服务。
    /// 实现应在全双工音频回调中<b>原地改写</b>麦克风采集(capture)信号，
    /// 滤除直流偏置(DC offset)与低频隆隆声(rumble)等低频成分。
    /// 通常置于采集处理链最前端（AEC/DNR 之前），以净化送入自适应滤波器的信号。
    /// </summary>
    public interface IHighPassFilterService : IDisposable
    {
        /// <summary>
        /// 处理一帧音频。
        /// </summary>
        /// <param name="capture">近端(麦克风)采集信号；实现应<b>原地改写</b>为高通滤波后的结果。</param>
        /// <param name="channels">声道数（交织排列）。</param>
        /// <param name="sampleRate">采样率 (Hz)。</param>
        void Process(Span<float> capture, uint channels, uint sampleRate);

        /// <summary>
        /// 重置内部状态（滤波器延迟线）。应在音频流停止/重启时调用，不应与 <see cref="Process"/> 并发。
        /// </summary>
        void Reset();
    }
}
