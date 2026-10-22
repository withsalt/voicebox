namespace VoiceBox.Core.Services.TTSService.Models
{
    /// <summary>
    /// 流式 TTS 输出的一段 PCM 数据。多通道时按帧交错排列 (frame-major, [s0c0, s0c1, s1c0, s1c1, ...])。
    /// </summary>
    public sealed class TtsAudioChunk
    {
        public required float[] Samples { get; init; }

        public required int SampleRate { get; init; }

        public required int Channels { get; init; }

        /// <summary>
        /// 距首段音频生成的累计时长 (秒)。
        /// </summary>
        public double EmittedSeconds { get; init; }

        /// <summary>
        /// 是否为本次合成的最后一块。
        /// </summary>
        public bool IsFinal { get; init; }
    }
}
