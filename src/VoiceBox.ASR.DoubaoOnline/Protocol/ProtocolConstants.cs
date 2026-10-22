namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// 火山引擎 WebSocket 二进制协议的常量。协议头固定 4 字节，整数字段一律大端。
    /// 详见文档「WebSocket 二进制协议」章节。
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
        /// <summary>端上发送的包含请求参数的 full client request。</summary>
        FullClientRequest = 0b0001,

        /// <summary>端上发送的包含音频数据的 audio only request。</summary>
        AudioOnlyRequest = 0b0010,

        /// <summary>服务端下发的包含识别结果的 full server response。</summary>
        FullServerResponse = 0b1001,

        /// <summary>服务端处理错误时下发的消息类型。</summary>
        ServerError = 0b1111,
    }

    /// <summary>Message type specific flags (header byte1 低 4 位)。</summary>
    internal enum MessageFlags : byte
    {
        /// <summary>header 后 4 字节不为 sequence number。</summary>
        NoSequence = 0b0000,

        /// <summary>header 后 4 字节为 sequence number 且为正。</summary>
        PositiveSequence = 0b0001,

        /// <summary>header 后 4 字节不为 sequence number，仅指示此为最后一包 (负包)。</summary>
        LastNoSequence = 0b0010,

        /// <summary>header 后 4 字节为 sequence number 且为负 (最后一包/负包)。</summary>
        NegativeSequence = 0b0011,
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
}
