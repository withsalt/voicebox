namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 文本片段。
    /// </summary>
    public sealed class TextContent : ChatContent
    {
        public TextContent(string text)
        {
            Text = text ?? string.Empty;
        }

        public string Text { get; }

        public override string ToString() => Text;
    }
}
