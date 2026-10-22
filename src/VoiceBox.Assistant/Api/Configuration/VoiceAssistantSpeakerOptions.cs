namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 本人说话人验证参数。在线身份认证由 <see cref="SpeakerConfirmationOptions.Enable"/> 统一开关：
    /// 关闭时即使安装了 Speaker provider 也不创建在线 verifier、身份 feed 或身份门；独立注册仍可使用 provider。
    /// 开启后，已安装的唤醒、播放期打断与连续对话单讲共用 <see cref="Confirmation"/> 的裁决参数。
    /// </summary>
    public sealed class VoiceAssistantSpeakerOptions
    {
        /// <summary>说话人服务的注册名。null 或空字符串表示使用默认说话人服务。</summary>
        public string? ServiceName { get; set; }

        /// <summary>
        /// 本人 profile（owner-speaker-profile-v2）文件。相对路径以 <see cref="AppContext.BaseDirectory"/> 为基准。
        /// </summary>
        public string ProfilePath { get; set; } = Path.Combine("tmp", "owner-speaker-v2.json");

        /// <summary>身份时间轴环形缓冲保留最近多少毫秒的身份 tap 音频。默认 6000ms。</summary>
        public int AudioRetentionMs { get; set; } = 6000;

        /// <summary>单 worker 推理调度器的有界任务数。默认 16。</summary>
        public int InferenceQueueCapacity { get; set; } = 16;

        /// <summary>允许用于实时 purpose（Wakeup/BargeInCommand）的 provider 最大窗长。默认 800ms。</summary>
        public int MaxRealtimeWindowMs { get; set; } = 800;

        /// <summary>语音门控预取最多覆盖最近多少个 hop 窗（在途 + 已缓存）。默认 4。</summary>
        public int PrefetchMaxWindows { get; set; } = 4;

        /// <summary>
        /// 身份 tap 上的短帧能量门。低于该 RMS 的样本不计入有效语音，不触发预取。默认 0.005。
        /// </summary>
        public float MinFrameRms { get; set; } = 0.005f;

        /// <summary>
        /// 可选 300ms 前缀高置信路径。默认关闭；仅在同时具备 owner 与非 owner 设备数据、且前缀高置信 FAR
        /// 不差于整窗高置信时才允许打开。打开后只走高置信 Match，不计普通票。
        /// </summary>
        public bool PrefixHighConfidenceEnabled { get; set; } = false;

        /// <summary>前缀高置信路径使用的语音起点后毫秒数，必须小于 provider 窗长。默认 300。</summary>
        public int PrefixHighConfidenceMs { get; set; } = 300;

        /// <summary>唤醒候选在 KWS focus 之后最多再收集多少毫秒证据。默认 300。</summary>
        public int WakeupPostTriggerEvidenceMs { get; set; } = 300;

        /// <summary>
        /// KWS 分数至少比其触发阈值高多少个百分点评为强唤醒证据。
        /// 强唤醒证据只把 required matches 降为 1，仍需当前候选至少一个普通说话人匹配。默认 15。
        /// </summary>
        public int WakeupStrongScoreMargin { get; set; } = 15;

        /// <summary>
        /// 各 purpose 共用的后备本人判定阈值。null 表示使用
        /// <see cref="Core.Services.SpeakerService.ISpeakerService.Threshold"/>。
        /// <see cref="SpeakerConfirmationOptions.Threshold"/> 优先于本项。
        /// </summary>
        public float? Threshold { get; set; }

        /// <summary>
        /// 唤醒、播放期打断与连续对话单讲共用的本人确认参数。是否生效只看
        /// <see cref="SpeakerConfirmationOptions.Enable"/>，再按已安装的场景取子集。
        /// </summary>
        public SpeakerConfirmationOptions Confirmation { get; set; } = new();

        internal string ResolveProfilePath()
            => Path.GetFullPath(Path.IsPathRooted(ProfilePath)
                ? ProfilePath
                : Path.Combine(AppContext.BaseDirectory, ProfilePath));
    }
}
