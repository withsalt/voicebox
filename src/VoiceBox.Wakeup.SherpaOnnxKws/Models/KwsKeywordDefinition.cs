namespace VoiceBox.Wakeup.SherpaOnnxKws.Models
{
    /// <summary>
    /// 单个唤醒词的运行时配置。可通过 <see cref="SherpaOnnxKwsOptions.AddKeyword"/> 添加,
    /// 也可以通过 <see cref="SherpaOnnxKwsOptions.KeywordsFile"/> 批量加载。
    /// </summary>
    public sealed class KwsKeywordDefinition
    {
        /// <summary>
        /// 关键词的原始文本 (用于 <see cref="Core.Services.WakeupService.Models.WakeupEventArgs.Keyword"/>)。
        /// 中文示例: "小爱同学"; 英文示例: "HELLO WORLD"。
        /// </summary>
        public string Keyword { get; set; } = string.Empty;

        /// <summary>
        /// 已经分词后的 token 序列, 与 tokens.txt 中的符号一一对应, 各 token 之间用空格分隔。
        /// <para>
        /// 中文 (cjkchar 模型): 单字之间空格, 例如 "小 爱 同 学";
        /// 英文 (bpe/char 模型): 与 tokens.txt 一致, 例如 "HH AH0 L OW1 W ER1 L D"。
        /// </para>
        /// </summary>
        public string Tokens { get; set; } = string.Empty;

        /// <summary>
        /// 该关键词的 boost score (大于 0 时生效, 用于上下文偏置加分; ≤ 0 表示沿用全局 KeywordsScore)。
        /// </summary>
        public float BoostScore { get; set; } = 0f;

        /// <summary>
        /// 该关键词单独的声学触发阈值, 范围 (0, 1] (大于 0 时生效; ≤ 0 表示沿用全局 KeywordsThreshold)。
        /// 阈值越高越严格 (更不易误唤醒)。
        /// </summary>
        public float Threshold { get; set; } = 0f;

        /// <summary>
        /// 该关键词命中后, 在 DebounceMs 时间内不再重复触发同名关键词。≤0 表示不限。
        /// </summary>
        public int DebounceMs { get; set; } = 800;

        /// <summary>
        /// 可选的发音变体原文。每条变体会被 AutoTokenizer 独立解析成 token 序列,
        /// 共享同一 <see cref="Keyword"/> 作为命中后回调的显示名。
        /// <para>
        /// 用途: 弥补声学模型对小语料关键词置信度低的问题。例如 "小爱同学" 常被
        /// 解成 "小爱通学" / "小爱同鞋", 可把这些误识结果作为变体注册,
        /// 但用户拿到的事件仍是 "小爱同学"。
        /// </para>
        /// 仅当 <see cref="Tokens"/> 为空 (即走 AutoTokenizer 路径) 时生效。
        /// </summary>
        public IList<string> Variants { get; } = new List<string>();
    }
}
