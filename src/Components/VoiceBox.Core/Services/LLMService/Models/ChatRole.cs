namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 多轮对话中的发言角色。
    /// </summary>
    public enum ChatRole
    {
        System = 0,
        User = 1,
        Assistant = 2,
        /// <summary>
        /// 结构化工具调用的结果。仅支持工具调用的 provider 可以消费该角色。
        /// </summary>
        Tool = 3,
    }
}
