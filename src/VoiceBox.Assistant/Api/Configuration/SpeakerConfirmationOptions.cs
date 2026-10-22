namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 唤醒、打断与单讲共用的本人确认裁决参数。
    /// 单讲低分证据在语音结束且身份音频到齐后才裁决拒绝；当前窗口满足匹配条件时可提前通过。
    /// </summary>
    public sealed class SpeakerConfirmationOptions
    {
        /// <summary>
        /// 是否开启在线本人确认。关闭时唤醒、播放期打断与单讲都不做说话人校验；
        /// 开启后，已安装的唤醒 / 播放期打断 / 连续对话单讲都走同一套确认参数。默认关闭。
        /// </summary>
        public bool Enable { get; set; }

        /// <summary>候选至少包含多少毫秒有效语音（按 speech mask 并集去重统计）。默认 500。</summary>
        public int MinimumSpeechMs { get; set; } = 500;

        /// <summary>裁决最多观察多少个 hop 窗。默认 3。</summary>
        public int WindowCount { get; set; } = 3;

        /// <summary>普通路径需要多少个匹配窗。默认 2/3。</summary>
        public int RequiredMatches { get; set; } = 2;

        /// <summary>普通阈值。null 依次使用 <see cref="VoiceAssistantSpeakerOptions.Threshold"/> 与 provider 阈值。</summary>
        public float? Threshold { get; set; }

        /// <summary>高置信单窗阈值。null 表示普通阈值 + 0.15（最大 1.0）。</summary>
        public float? HighConfidenceThreshold { get; set; }

        /// <summary>分数低于普通阈值但落在该差值内时记为临界证据，而不是明确 mismatch。默认 0.05。</summary>
        public float IndeterminateScoreMargin { get; set; } = 0.05f;

        /// <summary>
        /// Maximum worker wait after the first complete window is available. A timeout returns
        /// Unavailable(InferenceTimeout). Defaults to 150 milliseconds.
        /// </summary>
        public int InferenceTimeoutMs { get; set; } = 150;

        /// <summary>
        /// Duration for which a recent owner match can combine with one current normal match.
        /// Zero disables the prior. Defaults to 5000 milliseconds.
        /// </summary>
        public int RecentOwnerPriorMs { get; set; } = 5000;

        /// <summary>
        /// 是否保留"强上游证据 + 当前普通窗"一票路径。默认开启；关闭前必须有 owner / 非 owner FAR 记录。
        /// </summary>
        public bool AllowStrongUpstreamOneVote { get; set; } = true;
    }
}
