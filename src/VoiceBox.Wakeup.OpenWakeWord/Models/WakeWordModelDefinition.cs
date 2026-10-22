namespace VoiceBox.Wakeup.OpenWakeWord.Models
{
    /// <summary>
    /// 单个唤醒词模型的配置定义。
    /// </summary>
    public sealed class WakeWordModelDefinition
    {
        /// <summary>
        /// 唤醒词关键字 (回调事件中用于区分不同的唤醒词)。
        /// 必填。
        /// </summary>
        public string Keyword { get; set; } = string.Empty;

        /// <summary>
        /// 唤醒词 ONNX 模型路径。必填。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 触发阈值, 范围 [0, 1]。模型预测分数超过该阈值时认为命中。默认 0.5。
        /// </summary>
        public float Threshold { get; set; } = 0.5f;

        /// <summary>
        /// 防抖时间 (毫秒)。命中后该时长内不再重复触发同一唤醒词。默认 1000ms。
        /// </summary>
        public int DebounceMs { get; set; } = 1000;

        /// <summary>
        /// 连续命中帧数要求 (patience)。
        /// >0 时需要在最近 N 帧中均超过阈值才会触发, 用于降低误唤醒率。0 表示禁用 (默认)。
        /// </summary>
        public int Patience { get; set; } = 0;
    }
}
