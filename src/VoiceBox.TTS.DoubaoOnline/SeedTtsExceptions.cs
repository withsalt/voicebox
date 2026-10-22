namespace VoiceBox.TTS.DoubaoOnline
{
    /// <summary>火山引擎在线 TTS 相关异常的基类。</summary>
    public class DoubaoOnlineTTSException : Exception
    {
        public DoubaoOnlineTTSException(string message) : base(message) { }
        public DoubaoOnlineTTSException(string message, Exception inner) : base(message, inner) { }
    }

    /// <summary>二进制协议解析错误 (帧格式非法 / 截断等)。</summary>
    public sealed class SeedTtsProtocolException : DoubaoOnlineTTSException
    {
        public SeedTtsProtocolException(string message) : base(message) { }
    }

    /// <summary>
    /// 服务端返回的业务错误 (error 帧，或 ConnectionFailed / SessionFailed 事件)。
    /// </summary>
    public sealed class SeedTtsServerException : DoubaoOnlineTTSException
    {
        /// <summary>服务端错误码。error 帧携带；由事件失败下发时可能为 0。</summary>
        public uint Code { get; }

        /// <summary>服务端返回的 logid (X-Tt-Logid)，用于排障。</summary>
        public string? LogId { get; }

        public SeedTtsServerException(uint code, string message, string? logId = null)
            : base($"SeedTTS server error {code}: {message}" + (logId is null ? "" : $" (logid={logId})"))
        {
            Code = code;
            LogId = logId;
        }
    }
}
