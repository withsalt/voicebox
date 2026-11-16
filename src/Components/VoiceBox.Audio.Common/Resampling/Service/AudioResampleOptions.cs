namespace VoiceBox.Audio.Common.Resampling.Service
{
    /// <summary>
    /// <see cref="AudioResampleService"/> 的配置。
    /// </summary>
    public sealed class AudioResampleOptions
    {
        /// <summary>目标采样率 (Hz)。默认 16000 (与采集/AEC/VAD/ASR 链路一致)。</summary>
        public int TargetSampleRate { get; set; } = 16000;

        /// <summary>目标声道数。默认 1 (单声道)。</summary>
        public int TargetChannels { get; set; } = 1;

        /// <summary>
        /// 来源采样率 (Hz)。仅当 <see cref="IAudioResampleService.Convert(ReadOnlySpan{float}, int, int)"/>
        /// 未给出有效来源采样率 (&lt;=0) 时作为兜底。null 表示不兜底 (此时必须由调用方逐块给出)。
        /// </summary>
        public int? SourceSampleRate { get; set; }

        /// <summary>来源声道数兜底，语义同 <see cref="SourceSampleRate"/>。</summary>
        public int? SourceChannels { get; set; }

        /// <summary>
        /// Concentus/SpeexDSP 重采样质量，范围 0-10。默认 10。
        /// </summary>
        public int Quality { get; set; } = 10;
    }
}
