namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 图像片段。1.0 版本仅作为占位接口存在, 具体实现见 release notes。
    /// </summary>
    public sealed class ImageContent : ChatContent
    {
        public ImageContent(string path)
        {
            Path = path;
        }

        public ImageContent(byte[] bytes)
        {
            Bytes = bytes;
        }

        public string? Path { get; }

        public byte[]? Bytes { get; }
    }
}
