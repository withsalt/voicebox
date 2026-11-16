namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// 解析后的服务端下行帧。依据 <see cref="Type"/> 区分是识别结果 (<see cref="ServerMessageType.FullResponse"/>)
    /// 还是错误帧 (<see cref="ServerMessageType.Error"/>)。
    /// </summary>
    internal readonly struct ServerMessage
    {
        /// <summary>消息类型。</summary>
        public ServerMessageType Type { get; init; }

        /// <summary>
        /// 服务端下发的序号 (若帧头 flags 指示携带 sequence)。负数表示最后一包。
        /// 当 <see cref="HasSequence"/> 为 false 时该值无意义。
        /// </summary>
        public int Sequence { get; init; }

        /// <summary>帧头 flags 是否指示携带 sequence 字段。</summary>
        public bool HasSequence { get; init; }

        /// <summary>flags 是否指示这是最后一包 (bit1 置位)。</summary>
        public bool IsLastPackage { get; init; }

        /// <summary>识别结果的 JSON 文本 (已解压)。仅 <see cref="ServerMessageType.FullResponse"/> 有效。</summary>
        public string? PayloadJson { get; init; }

        /// <summary>错误码。仅 <see cref="ServerMessageType.Error"/> 有效。</summary>
        public uint ErrorCode { get; init; }

        /// <summary>错误信息。仅 <see cref="ServerMessageType.Error"/> 有效。</summary>
        public string? ErrorMessage { get; init; }
    }

    internal enum ServerMessageType
    {
        FullResponse,
        Error,
    }
}
