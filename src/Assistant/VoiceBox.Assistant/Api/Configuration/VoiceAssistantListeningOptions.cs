namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 语音助手 Idle/连续对话态监听确认参数。
    /// </summary>
    public sealed class VoiceAssistantListeningOptions
    {
        /// <summary>
        /// Idle (含连续对话窗口) 状态下, 新语音候选需累计至少该时长的音频才确认开始一轮对话。
        /// 确认前不进入 Listening、不启动 ASR 会话、不刷新连续对话活跃时间, 用于过滤
        /// TTS 回声残差/环境噪声触发的 VAD 误报。与 <see cref="MinRms"/> 同时为 0
        /// 时关闭该门槛 (旧行为: VAD 一触发即开始)。默认 200ms。
        /// <para>注意: 时长按 VAD 外发音频计量, 其首块通常已含 ~300ms 前置 padding
        /// (FireRedVAD 默认), 因此该门槛的实际过滤主力是 <see cref="MinRms"/>;
        /// 时长换算使用音频服务配置的采样率和声道数。</para>
        /// </summary>
        public int MinDurationMs { get; set; } = 200;

        /// <summary>
        /// Idle 状态下确认新一轮对话所需的最小累积 RMS。低于该值的候选会被静默丢弃
        /// (仅 Debug 日志, 无对外事件)。默认 0.02 (约 -34 dBFS)。设为 0 表示只按时长门槛确认。
        /// <para>注意: RMS 按 VAD 外发音频 (含前置 padding 静音) 累积计量, 比纯语音段的 RMS 偏低:
        /// 经 AEC 后的 TTS 回声残差在此口径下实测约 0.014 (被默认值拦截);
        /// 正常距离人声的首块合成 RMS 一般 >=0.027, 可通过; 远场/耳语若整段低于 0.02
        /// 会被整句丢弃 - 此类场景请调低该值或置 0。</para>
        /// </summary>
        public float MinRms { get; set; } = 0.02f;
    }
}
