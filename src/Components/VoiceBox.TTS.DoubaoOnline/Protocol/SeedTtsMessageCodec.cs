using System.Buffers.Binary;
using System.Text;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// 火山引擎双向流式 TTS「事件版」二进制协议的编解码器。负责：
    ///   * 构造带事件的 full client request 帧 (4 字节头 + event + 可选 session_id + payload)；
    ///   * 解析服务端下行帧 (full server response / audio only server / error)。
    /// 该类为纯函数、无状态，便于单元测试逐字节校验。
    /// </summary>
    internal static class SeedTtsMessageCodec
    {
        /// <summary>
        /// 构造一个带事件的 full client request 帧。<paramref name="sessionId"/> 对连接级事件
        /// (StartConnection/FinishConnection) 传 null 即可，编码时会自动省略 session_id 字段。
        /// </summary>
        public static byte[] EncodeClientEvent(EventType evt, string? sessionId, string json, bool gzip)
        {
            byte[] payload = Encoding.UTF8.GetBytes(json);
            if (gzip) payload = GzipCodec.Compress(payload);

            bool omitSession = evt.OmitsSessionId();
            byte[]? sidBytes = omitSession ? null : Encoding.UTF8.GetBytes(sessionId ?? string.Empty);

            int size = ProtocolConstants.HeaderBytes
                       + 4                                    // event
                       + (omitSession ? 0 : 4 + sidBytes!.Length) // session_id_size + session_id
                       + 4                                    // payload_size
                       + payload.Length;

            var frame = new byte[size];
            frame[0] = (byte)((ProtocolConstants.ProtocolVersion1 << 4) | ProtocolConstants.HeaderSize4);
            frame[1] = (byte)(((byte)MessageType.FullClientRequest << 4) | (byte)MessageFlags.WithEvent);
            frame[2] = (byte)(((byte)SerializationMethod.Json << 4) | (byte)(gzip ? CompressionMethod.Gzip : CompressionMethod.None));
            frame[3] = 0x00; // reserved

            int offset = ProtocolConstants.HeaderBytes;
            BinaryPrimitives.WriteInt32BigEndian(frame.AsSpan(offset, 4), (int)evt);
            offset += 4;

            if (!omitSession)
            {
                BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(offset, 4), (uint)sidBytes!.Length);
                offset += 4;
                sidBytes.CopyTo(frame, offset);
                offset += sidBytes.Length;
            }

            BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(offset, 4), (uint)payload.Length);
            offset += 4;
            payload.CopyTo(frame, offset);
            return frame;
        }

        /// <summary>
        /// 解析一条完整的服务端下行消息。<paramref name="frame"/> 必须是一整个 WebSocket 消息
        /// (已完成分片重组)。解析失败会抛出 <see cref="SeedTtsProtocolException"/>。
        /// </summary>
        public static ServerMessage Decode(ReadOnlySpan<byte> frame)
        {
            if (frame.Length < ProtocolConstants.HeaderBytes)
                throw new SeedTtsProtocolException($"Frame too short: {frame.Length} bytes.");

            int headerSize = (frame[0] & 0x0F) * 4;
            if (headerSize < ProtocolConstants.HeaderBytes || headerSize > frame.Length)
                throw new SeedTtsProtocolException($"Invalid header size: {headerSize}.");

            var messageType = (MessageType)(frame[1] >> 4);
            byte flags = (byte)(frame[1] & 0x0F);
            var compression = (CompressionMethod)(frame[2] & 0x0F);
            bool hasEvent = (flags & (byte)MessageFlags.WithEvent) != 0;

            int offset = headerSize;

            if (messageType == MessageType.Error)
                return DecodeError(frame, ref offset, compression);

            var evt = EventType.None;
            string? sessionId = null;
            string? connectId = null;

            if (hasEvent)
            {
                EnsureRemaining(frame, offset, 4, "event");
                evt = (EventType)BinaryPrimitives.ReadInt32BigEndian(frame.Slice(offset, 4));
                offset += 4;

                if (!evt.OmitsSessionId())
                    sessionId = ReadLengthPrefixedString(frame, ref offset, "session_id");

                if (evt.HasConnectId())
                    connectId = ReadLengthPrefixedString(frame, ref offset, "connect_id");
            }

            byte[] payload = ReadLengthPrefixedPayload(frame, ref offset, compression, "payload");

            if (messageType == MessageType.AudioOnlyServer)
            {
                return new ServerMessage
                {
                    Type = messageType,
                    Event = evt,
                    SessionId = sessionId,
                    ConnectId = connectId,
                    AudioPayload = payload,
                };
            }

            return new ServerMessage
            {
                Type = messageType,
                Event = evt,
                SessionId = sessionId,
                ConnectId = connectId,
                Json = payload.Length == 0 ? string.Empty : Encoding.UTF8.GetString(payload),
            };
        }

        private static ServerMessage DecodeError(ReadOnlySpan<byte> frame, ref int offset, CompressionMethod compression)
        {
            EnsureRemaining(frame, offset, 4, "error code");
            uint code = BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;

            byte[] payload = ReadLengthPrefixedPayload(frame, ref offset, compression, "error message");
            return new ServerMessage
            {
                Type = MessageType.Error,
                Event = EventType.None,
                ErrorCode = code,
                ErrorMessage = payload.Length == 0 ? string.Empty : Encoding.UTF8.GetString(payload),
            };
        }

        private static string ReadLengthPrefixedString(ReadOnlySpan<byte> frame, ref int offset, string what)
        {
            EnsureRemaining(frame, offset, 4, $"{what} size");
            int size = (int)BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;
            if (size < 0) throw new SeedTtsProtocolException($"Negative {what} size: {size}.");
            EnsureRemaining(frame, offset, size, what);
            string s = size == 0 ? string.Empty : Encoding.UTF8.GetString(frame.Slice(offset, size));
            offset += size;
            return s;
        }

        private static byte[] ReadLengthPrefixedPayload(
            ReadOnlySpan<byte> frame, ref int offset, CompressionMethod compression, string what)
        {
            EnsureRemaining(frame, offset, 4, $"{what} size");
            int size = (int)BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;
            if (size < 0) throw new SeedTtsProtocolException($"Negative {what} size: {size}.");
            EnsureRemaining(frame, offset, size, what);

            ReadOnlySpan<byte> raw = frame.Slice(offset, size);
            offset += size;
            if (size == 0) return Array.Empty<byte>();
            return compression == CompressionMethod.Gzip ? GzipCodec.Decompress(raw) : raw.ToArray();
        }

        private static void EnsureRemaining(ReadOnlySpan<byte> frame, int offset, int need, string what)
        {
            // 用减法而非 offset+need 比较, 避免超大 size 导致 int 溢出绕过校验 (offset 始终 ≤ frame.Length)。
            if (need > frame.Length - offset)
                throw new SeedTtsProtocolException(
                    $"Truncated frame: need {need} bytes for {what} at offset {offset}, have {frame.Length - offset}.");
        }
    }
}
