namespace VoiceBox.Audio.Common.Resampling.Service
{
    /// <summary>
    /// 自适应、高保真的音频采样率 + 声道数转换服务 (仅供 VoiceBox.Assistant 内部使用)。
    ///
    /// 典型场景：TTS 输出 (如 24kHz/1ch、48kHz/2ch) → 播放/AEC 链路要求的格式 (通常 16kHz/1ch)。
    /// 用 Concentus/SpeexDSP 重采样器实现，下采样时抑制混叠；按 chunk 流式转换并跨块保持相位。
    ///
    /// 来源格式可由调用方在 <see cref="Convert(ReadOnlySpan{float}, int, int)"/> 时显式给出
    /// (例如从 TTS chunk 自描述格式自动检测)；检测不到时回落到构造时配置的来源格式。
    ///
    /// 有状态：实例维护重采样相位与历史，<b>不是线程安全的单流抽象</b>，
    /// 不应跨并发流共享 (实现内部加锁仅为防御性保护，不代表支持并发流)。
    /// </summary>
    public interface IAudioResampleService : IDisposable
    {
        /// <summary>目标采样率 (Hz)。</summary>
        int TargetSampleRate { get; }

        /// <summary>目标声道数。</summary>
        int TargetChannels { get; }

        /// <summary>
        /// 转换一段交错 PCM 到目标格式，结果写入<b>调用方持有</b>的 <paramref name="output"/> 缓冲
        /// (按需扩容，无需预填)，返回写入的样本数。零分配热路径：调用方跨块复用同一缓冲。
        /// <paramref name="sourceSampleRate"/>/<paramref name="sourceChannels"/> &lt;= 0 时回落到配置的
        /// 来源格式；若也未配置则抛 <see cref="InvalidOperationException"/>。
        /// </summary>
        int Convert(ReadOnlySpan<float> samples, int sourceSampleRate, int sourceChannels, ref float[] output);

        /// <summary>冲洗群延迟尾部到 <paramref name="output"/> (按需扩容)，返回写入的样本数。</summary>
        int Flush(ref float[] output);

        /// <summary>
        /// 转换一段交错 PCM 到目标格式 (分配新数组返回，便于一次性调用)。
        /// <paramref name="sourceSampleRate"/>/<paramref name="sourceChannels"/> &lt;= 0 时，
        /// 回落到配置的来源格式；若也未配置则抛 <see cref="InvalidOperationException"/>。
        /// </summary>
        float[] Convert(ReadOnlySpan<float> samples, int sourceSampleRate, int sourceChannels);

        /// <summary>用配置的来源格式转换 (适用于来源格式固定、调用方不逐块描述的场景)。</summary>
        float[] Convert(ReadOnlySpan<float> samples);

        /// <summary>
        /// 冲洗重采样器群延迟里尚未输出的尾部样本 (流/句结束时调用，避免截断句尾，分配新数组返回)。
        /// </summary>
        float[] Flush();

        /// <summary>复位重采样相位与历史，开始一个新的独立流。不清除已构建的滤波器表。</summary>
        void Reset();
    }
}
