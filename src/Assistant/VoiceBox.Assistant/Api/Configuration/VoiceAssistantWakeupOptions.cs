namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 语音助手唤醒词门控参数。
    /// </summary>
    public sealed class VoiceAssistantWakeupOptions
    {
        /// <summary>
        /// 是否启用唤醒词门控。启用后, 麦克风音频在休眠状态只送入 <see cref="Core.Services.WakeupService.IWakeupService"/>,
        /// 命中唤醒词后才进入 VAD/ASR 对话链路。连续对话窗口内无需再次唤醒。
        /// 在线本人确认由 <see cref="VoiceAssistantSpeakerOptions.Confirmation"/> 的
        /// <see cref="SpeakerConfirmationOptions.Enable"/> 统一控制，与本开关独立。
        /// </summary>
        public bool Enable { get; set; } = false;

        /// <summary>
        /// 唤醒词服务的注册名。null 或空字符串表示使用默认 <see cref="Core.Services.WakeupService.IWakeupService"/>。
        /// </summary>
        public string? ServiceName { get; set; }

        /// <summary>
        /// 休眠期间保留多少毫秒最近麦克风音频。唤醒命中后会把这段音频补送给 VAD,
        /// 支持"唤醒词 + 后续命令"连在一句里说。默认 2500ms。
        /// </summary>
        public int PreRollMs { get; set; } = 2500;

        /// <summary>
        /// 唤醒命中后补送给 VAD 的音频从"估算唤醒词开始点"前多少毫秒开始。
        /// KWS 的开始点通常来自帧级/token 峰值估计, 不一定是声学起音点; 默认 400ms
        /// 给中文短唤醒词留足余量, 避免 ASR 偶发少掉唤醒词开头。
        /// </summary>
        public int ReplayLeadInMs { get; set; } = 400;
    }
}
