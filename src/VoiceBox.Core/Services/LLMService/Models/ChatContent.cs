namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 多模态消息内容片段的基类。文本/音频/图像各有具体派生类。
    /// 通过派生类型区分而非字符串枚举，避免字段名拼写错误并便于在编译期获得分支覆盖检查。
    /// </summary>
    public abstract class ChatContent
    {
        private protected ChatContent() { }
    }

}
