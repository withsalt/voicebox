namespace VoiceBox.Core.Services.AudioService.Models
{
    public abstract class AudioOptionsBase
    {
        /// <summary>
        /// 音频服务采样率 (Hz)。
        /// 默认值: 16000。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 音频服务声道数。
        /// 默认值: 1 (单声道)。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// 音频服务后端类型。
        /// </summary>
        public BackendType BackendType { get; set; } = BackendType.PortAudio;
    }
}
