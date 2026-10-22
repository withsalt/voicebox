namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>火山引擎 ASR 相关异常的基类。</summary>
    public class DoubaoOnlineASRException : Exception
    {
        public DoubaoOnlineASRException(string message) : base(message) { }
        public DoubaoOnlineASRException(string message, Exception inner) : base(message, inner) { }
    }

    /// <summary>二进制协议解析错误 (帧格式非法 / 截断等)。</summary>
    public sealed class SeedAsrProtocolException : DoubaoOnlineASRException
    {
        public SeedAsrProtocolException(string message) : base(message) { }
    }

    /// <summary>
    /// 本地实时音频生产速度长期超过网络发送速度，继续接收将无法在不丢句首的前提下
    /// 保持有界内存。该异常表示本次识别音频不完整，调用方应放弃结果并重试。
    /// </summary>
    public sealed class SeedAsrAudioBufferOverflowException : DoubaoOnlineASRException
    {
        public string SessionId { get; }
        public long CapacitySamples { get; }
        public long BufferedSamples { get; }
        public int RejectedSamples { get; }

        public SeedAsrAudioBufferOverflowException(
            string sessionId,
            long capacitySamples,
            long bufferedSamples,
            int rejectedSamples)
            : base(
                $"SeedASR session {sessionId} audio buffer overflow: " +
                $"buffered={bufferedSamples}, rejected={rejectedSamples}, " +
                $"capacity={capacitySamples} samples. No queued audio was dropped.")
        {
            SessionId = sessionId;
            CapacitySamples = capacitySamples;
            BufferedSamples = bufferedSamples;
            RejectedSamples = rejectedSamples;
        }
    }

    /// <summary>
    /// 服务端返回的业务错误 (error 帧或 payload 中的非成功错误码)。
    /// </summary>
    public sealed class SeedAsrServerException : DoubaoOnlineASRException
    {
        /// <summary>服务端错误码，详见文档「错误码」章节。</summary>
        public uint Code { get; }

        /// <summary>服务端返回的 logid (X-Tt-Logid)，用于排障。</summary>
        public string? LogId { get; }

        public SeedAsrServerException(uint code, string message, string? logId = null)
            : base($"SeedASR server error {code}: {message}" + (logId is null ? "" : $" (logid={logId})"))
        {
            Code = code;
            LogId = logId;
        }
    }
}
