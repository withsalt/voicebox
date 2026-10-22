using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 语音助手对话行为参数。
    /// </summary>
    public sealed class VoiceAssistantConversationOptions
    {
        /// <summary>
        /// 是否允许一次唤醒后连续多轮对话。关闭后, 每个回合结束都会立即回到待唤醒状态;
        /// 回合进行中仍可用唤醒词打断当前回复。
        /// </summary>
        public bool ContinuousConversationEnabled { get; set; } = true;

        /// <summary>
        /// 连续对话模式下, 从最近一次交互 (确认的用户语音或助手回合完成) 起计算,
        /// 超过该时长没有新交互则回到待唤醒。超时只在真正空闲时判定:
        /// 回合进行中 (Listening/Recognizing/Thinking/Speaking) 或语音候选累积期间不会过期。
        /// 默认 60 秒。
        /// </summary>
        public TimeSpan ContinuousConversationIdleTimeout { get; set; } = TimeSpan.FromSeconds(30);

        /// <summary>
        /// 启用唤醒词门控并处于连续对话模式时用于结束当前对话的词汇。识别文本包含其中任一词汇时,
        /// 当前回合不会提交给 LLM, 而是回到待唤醒状态。匹配时会忽略空白和标点。
        /// 清空此列表可关闭结束词功能。
        /// </summary>
        public IList<string> EndConversationKeywords { get; set; } = new List<string>
        {
            "再见",
            "拜拜",
            "关机",
            "下次聊",
            "下次再聊",
            "结束对话",
            "停一下",
            "停下来",
        };

        /// <summary>
        /// LLM 是否启用 thinking 输出。启用后, thinking 块会通过
        /// <see cref="AssistantTextEventArgs.IsThinking"/> 对外展示, 但不会进入 TTS 或对话历史。
        /// </summary>
        public bool EnableThinking { get; set; } = false;
    }
}
