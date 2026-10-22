namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// 火山引擎双向流式 TTS WebSocket 二进制协议常量。协议头固定 4 字节；整数字段一律大端。
    /// 与 ASR 的「序号版」不同，本协议为「事件版」：所有帧都携带 <see cref="MessageFlags.WithEvent"/>，
    /// header 之后紧跟 4 字节 event，再按事件类型可选携带 session_id / connect_id，最后是 payload。
    /// 详见文档「双向流式语音合成 WebSocket」及官方 protocols_.py。
    /// </summary>
    internal static class ProtocolConstants
    {
        /// <summary>协议版本，目前只有 version 1。</summary>
        public const byte ProtocolVersion1 = 0b0001;

        /// <summary>Header 大小 (以 4 字节为单位)，1 表示 header = 4 字节。</summary>
        public const byte HeaderSize4 = 0b0001;

        /// <summary>协议头字节数 = HeaderSize4 * 4。</summary>
        public const int HeaderBytes = 4;
    }

    /// <summary>消息类型 (header byte1 高 4 位)。</summary>
    internal enum MessageType : byte
    {
        /// <summary>端上发送的包含事件 + JSON 的 full client request。</summary>
        FullClientRequest = 0b0001,

        /// <summary>端上发送的仅含音频的 audio only client request (本 TTS 流程不用)。</summary>
        AudioOnlyClient = 0b0010,

        /// <summary>服务端下发的包含事件 + JSON 的 full server response。</summary>
        FullServerResponse = 0b1001,

        /// <summary>服务端下发的仅含音频的 audio only server response (TTSResponse 音频块)。</summary>
        AudioOnlyServer = 0b1011,

        /// <summary>服务端处理错误时下发的消息类型。</summary>
        Error = 0b1111,
    }

    /// <summary>Message type specific flags (header byte1 低 4 位)。</summary>
    internal enum MessageFlags : byte
    {
        /// <summary>无序号、无事件。</summary>
        NoSequence = 0b0000,

        /// <summary>携带正序号。</summary>
        PositiveSequence = 0b0001,

        /// <summary>无序号，仅指示最后一包。</summary>
        LastNoSequence = 0b0010,

        /// <summary>携带负序号 (最后一包)。</summary>
        NegativeSequence = 0b0011,

        /// <summary>携带事件号 (header 后 4 字节为 event)。双向流式 TTS 全程使用该标志。</summary>
        WithEvent = 0b0100,
    }

    /// <summary>Payload 序列化方式 (header byte2 高 4 位)。</summary>
    internal enum SerializationMethod : byte
    {
        None = 0b0000,
        Json = 0b0001,
    }

    /// <summary>Payload 压缩方式 (header byte2 低 4 位)。</summary>
    internal enum CompressionMethod : byte
    {
        None = 0b0000,
        Gzip = 0b0001,
    }

    /// <summary>
    /// 事件类型 (header 后 4 字节，大端有符号 int32)。取值经官方 protocols_.py 校验。
    /// 注意 TTS 段不连续：TTSSentenceStart/End/Response=350/351/352，TTSEnded=359，
    /// 而 <see cref="TtsSubtitle"/> = 364 (并非 353)。
    /// </summary>
    internal enum EventType
    {
        None = 0,

        // ---- 上行：连接类 ----
        StartConnection = 1,
        FinishConnection = 2,

        // ---- 下行：连接类 ----
        ConnectionStarted = 50,
        ConnectionFailed = 51,
        ConnectionFinished = 52,

        // ---- 上行：会话类 ----
        StartSession = 100,
        CancelSession = 101,
        FinishSession = 102,

        // ---- 下行：会话类 ----
        SessionStarted = 150,
        SessionCanceled = 151,
        SessionFinished = 152,
        SessionFailed = 153,
        UsageResponse = 154,

        // ---- 上行：通用 ----
        TaskRequest = 200,

        // ---- 下行：TTS ----
        TtsSentenceStart = 350,
        TtsSentenceEnd = 351,
        TtsResponse = 352,
        TtsEnded = 359,
        TtsSubtitle = 364,
    }

    internal static class EventTypeExtensions
    {
        /// <summary>
        /// 该事件在「写/读」路径上是否<b>省略</b> session_id 字段。连接级事件不带 session_id。
        /// 读路径的省略集比写路径多一个 <see cref="EventType.ConnectionFinished"/>。
        /// </summary>
        public static bool OmitsSessionId(this EventType e) => e is
            EventType.StartConnection or
            EventType.FinishConnection or
            EventType.ConnectionStarted or
            EventType.ConnectionFailed or
            EventType.ConnectionFinished;

        /// <summary>该事件在读路径上是否携带 connect_id 字段 (仅连接级下行事件)。</summary>
        public static bool HasConnectId(this EventType e) => e is
            EventType.ConnectionStarted or
            EventType.ConnectionFailed or
            EventType.ConnectionFinished;
    }
}
