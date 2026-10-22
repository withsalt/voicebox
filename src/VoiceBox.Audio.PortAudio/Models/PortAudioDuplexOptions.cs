using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio.Models
{
    public class PortAudioDuplexOptions : AudioOptionsBase
    {
        /// <summary>
        /// 录制队列最大堆积数量，当堆积超过此数量时，将丢弃旧数据。
        /// </summary>
        public uint RecordBufferCapacity { get; set; } = 100;

        /// <summary>
        /// 播放队列大小，队列堆积超过此大小后，音频数据将被丢弃。
        /// </summary>
        public uint PlayMaxQueueSize { get; set; } = 300;

        /// <summary>
        /// 三路/四路全双工音频帧快照队列容量。仅当订阅 OnDuplexAudioFrame 时使用；
        /// 队列满时丢弃旧帧，避免阻塞实时音频回调。
        /// </summary>
        public uint DuplexFrameBufferCapacity { get; set; } = 128;

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

        /// <summary>
        /// 是否启用声学回声消除(AEC)。启用后需要在依赖注入容器中注册
        /// IAcousticEchoCancellationService 实现。
        /// </summary>
        public bool EnableAec { get; set; } = false;

        /// <summary>
        /// 是否启用数字降噪(DNR)。启用后需要在依赖注入容器中注册
        /// IDigitalNoiseReductionService 实现。
        /// </summary>
        public bool EnableDnr { get; set; } = false;

        /// <summary>
        /// 是否启用高通滤波(HPF)。启用后需要在依赖注入容器中注册
        /// IHighPassFilterService 实现。位于采集处理链最前端（AEC/DNR 之前），
        /// 用于滤除直流偏置与低频隆隆声。
        /// </summary>
        public bool EnableHpf { get; set; } = false;
    }
}
