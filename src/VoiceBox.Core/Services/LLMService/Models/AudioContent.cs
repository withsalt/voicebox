namespace VoiceBox.Core.Services.LLMService.Models
{
    /// <summary>
    /// 音频片段。同时支持文件路径或内存中的原始字节 (例如 WAV/MP3 byte[])。
    /// 仅设置其中之一；同时设置时以 <see cref="Bytes"/> 优先。
    /// </summary>
    public sealed class AudioContent : ChatContent
    {
        public AudioContent(string path)
        {
            Path = path;
        }

        public AudioContent(byte[] bytes)
        {
            Bytes = bytes;
        }

        /// <summary>
        /// 本地音频文件路径。
        /// </summary>
        public string? Path { get; }

        /// <summary>
        /// 音频文件原始字节 (含容器头, 如 WAV/MP3)。
        /// </summary>
        public byte[]? Bytes { get; }
    }
}
