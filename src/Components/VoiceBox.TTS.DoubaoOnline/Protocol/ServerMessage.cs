namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// 解析后的服务端下行帧。依据 <see cref="Type"/> 与 <see cref="Event"/> 区分内容：
    /// 音频块 (<see cref="MessageType.AudioOnlyServer"/> + <see cref="EventType.TtsResponse"/>) 的
    /// <see cref="AudioPayload"/> 为原始 PCM 字节；其余事件 payload 为 JSON (见 <see cref="Json"/>)；
    /// error 帧携带 <see cref="ErrorCode"/> 与 <see cref="ErrorMessage"/>。
    /// </summary>
    internal readonly struct ServerMessage
    {
        /// <summary>消息类型 (帧头 byte1 高 4 位)。</summary>
        public MessageType Type { get; init; }

        /// <summary>事件号 (WithEvent 帧携带)。非事件帧为 <see cref="EventType.None"/>。</summary>
        public EventType Event { get; init; }

        /// <summary>会话 ID (存在时)。</summary>
        public string? SessionId { get; init; }

        /// <summary>连接 ID (连接级下行事件携带)。</summary>
        public string? ConnectId { get; init; }

        /// <summary>非音频事件的 JSON payload (已解压)。</summary>
        public string? Json { get; init; }

        /// <summary>音频事件的原始 PCM 字节 (已解压)。</summary>
        public byte[]? AudioPayload { get; init; }

        /// <summary>错误码。仅 <see cref="MessageType.Error"/> 有效。</summary>
        public uint ErrorCode { get; init; }

        /// <summary>错误信息。仅 <see cref="MessageType.Error"/> 有效。</summary>
        public string? ErrorMessage { get; init; }

        public bool IsAudio => Type == MessageType.AudioOnlyServer;
        public bool IsError => Type == MessageType.Error;
    }
}
