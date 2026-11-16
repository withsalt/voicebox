using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Models
{
    public class PortAudioRecordOptions : AudioOptionsBase
    {
        /// <summary>
        /// 录制队列最大堆积数量，当堆积超过此数量时，将丢弃旧数据。
        /// </summary>
        public uint BufferCapacity { get; set; } = 100;

        /// <summary>
        /// 建议延迟（秒）。null 表示使用设备默认低延迟值。
        /// 较低的值可减少延迟但可能导致音频断续（xrun）。
        /// 典型值：0.005-0.010（极低延迟），0.020-0.050（低延迟）
        /// </summary>
        public double? SuggestedLatency { get; set; } = null;

        /// <summary>
        /// 最大帧大小
        /// 由音频服务根据采样率计算（-1），由PortAudio自决定（0），其他值则为指定的帧大小。
        /// </summary>
        public int MaxFrameSize { get; set; } = -1;

        /// <summary>
        /// 启用 WASAPI 独占模式（仅 Windows）。
        /// 独占模式可提供最低延迟，但会独占音频设备。
        /// 仅当 Host API 为 WASAPI 时有效。
        /// </summary>
        public bool EnableWasapiExclusiveMode { get; set; } = false;
    }
}
