using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Models
{
    public class PortAudioPlayOptions : AudioOptionsBase
    {
        /// <summary>
        /// 缓存队列大小，队列堆积超过此大小后，音频数据将被丢弃。
        /// </summary>
        public uint MaxQueueSize { get; set; } = 300;

        /// <summary>
        /// 最大帧大小
        /// 由音频服务根据采样率计算（-1），由PortAudio自决定（0），其他值则为指定的帧大小。
        /// </summary>
        public int MaxFrameSize { get; set; } = -1;

        /// <summary>
        /// 建议延迟（秒）。null 表示使用设备默认低延迟值。
        /// 较低的值可减少延迟但可能导致音频断续（xrun）。
        /// 典型值：0.005-0.010（极低延迟），0.020-0.050（低延迟）
        /// </summary>
        public double? SuggestedLatency { get; set; } = null;

        /// <summary>
        /// 启用 WASAPI 独占模式（仅 Windows）。
        /// 独占模式可提供最低延迟，但会独占音频设备。
        /// 仅当 Host API 为 WASAPI 时有效。
        /// </summary>
        public bool EnableWasapiExclusiveMode { get; set; } = false;
    }
}
