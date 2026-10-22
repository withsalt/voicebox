using System.Buffers.Binary;
using System.Text;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// 火山引擎 WebSocket 二进制协议的编解码器。负责：
    ///   * 构造 full client request / audio only request 帧 (4 字节头 + 大端 payload size + payload)；
    ///   * 解析 full server response / error 帧。
    /// 该类为纯函数、无状态，便于单元测试逐字节校验。
    /// </summary>
    internal static class SeedAsrMessageCodec
    {
        /// <summary>
        /// 构造 full client request 帧。payload 为 JSON (可选 Gzip 压缩)。
        /// 客户端首包不携带 sequence，使用 <see cref="MessageFlags.NoSequence"/>。
        /// </summary>
        public static byte[] EncodeFullClientRequest(string json, bool gzip)
        {
            byte[] payload = Encoding.UTF8.GetBytes(json);
            if (gzip) payload = GzipCodec.Compress(payload);
            return BuildFrame(
                MessageType.FullClientRequest,
                MessageFlags.NoSequence,
                SerializationMethod.Json,
                gzip ? CompressionMethod.Gzip : CompressionMethod.None,
                payload);
        }

        /// <summary>
        /// 构造 audio only request 帧。<paramref name="isLast"/> 为 true 时标记为最后一包 (负包)。
        /// 音频负载为原始字节 (无序列化)，可选 Gzip 压缩。
        /// </summary>
        public static byte[] EncodeAudioRequest(ReadOnlySpan<byte> audio, bool gzip, bool isLast)
        {
            // 空载荷（最后一包且无剩余音频）绝不压缩：gzip 空字节流在某些版本的 .NET 下写出不完整的
            // gzip header，服务端解压时直接报 EOF。压缩标志必须与实际载荷匹配。
            bool actuallyGzip = gzip && !audio.IsEmpty;
            byte[] payload = actuallyGzip ? GzipCodec.Compress(audio) : audio.ToArray();
            return BuildFrame(
                MessageType.AudioOnlyRequest,
                isLast ? MessageFlags.LastNoSequence : MessageFlags.NoSequence,
                SerializationMethod.None,
                actuallyGzip ? CompressionMethod.Gzip : CompressionMethod.None,
                payload);
        }

        private static byte[] BuildFrame(
            MessageType type,
            MessageFlags flags,
            SerializationMethod serialization,
            CompressionMethod compression,
            byte[] payload)
        {
            var frame = new byte[ProtocolConstants.HeaderBytes + 4 + payload.Length];
            frame[0] = (byte)((ProtocolConstants.ProtocolVersion1 << 4) | ProtocolConstants.HeaderSize4);
            frame[1] = (byte)(((byte)type << 4) | (byte)flags);
            frame[2] = (byte)(((byte)serialization << 4) | (byte)compression);
            frame[3] = 0x00; // reserved
            BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(4, 4), (uint)payload.Length);
            payload.CopyTo(frame, ProtocolConstants.HeaderBytes + 4);
            return frame;
        }

        /// <summary>
        /// 解析一条完整的服务端下行消息。<paramref name="frame"/> 必须是一整个 WebSocket 消息
        /// (已完成分片重组)。解析失败会抛出 <see cref="SeedAsrProtocolException"/>。
        /// </summary>
        public static ServerMessage Decode(ReadOnlySpan<byte> frame)
        {
            if (frame.Length < ProtocolConstants.HeaderBytes)
                throw new SeedAsrProtocolException($"Frame too short: {frame.Length} bytes.");

            int headerSize = (frame[0] & 0x0F) * 4;
            if (headerSize < ProtocolConstants.HeaderBytes || headerSize > frame.Length)
                throw new SeedAsrProtocolException($"Invalid header size: {headerSize}.");

            var messageType = (MessageType)(frame[1] >> 4);
            byte flags = (byte)(frame[1] & 0x0F);
            var compression = (CompressionMethod)(frame[2] & 0x0F);

            bool hasSequence = (flags & 0b0001) != 0;
            bool isLast = (flags & 0b0010) != 0;

            int offset = headerSize;
            int sequence = 0;
            if (hasSequence)
            {
                EnsureRemaining(frame, offset, 4, "sequence");
                sequence = BinaryPrimitives.ReadInt32BigEndian(frame.Slice(offset, 4));
                offset += 4;
            }

            return messageType switch
            {
                MessageType.FullServerResponse => DecodeResponse(frame, offset, compression, sequence, hasSequence, isLast),
                MessageType.ServerError => DecodeError(frame, offset, compression),
                _ => throw new SeedAsrProtocolException($"Unexpected server message type: 0b{Convert.ToString((byte)messageType, 2).PadLeft(4, '0')}."),
            };
        }

        private static ServerMessage DecodeResponse(
            ReadOnlySpan<byte> frame, int offset, CompressionMethod compression,
            int sequence, bool hasSequence, bool isLast)
        {
            EnsureRemaining(frame, offset, 4, "payload size");
            int size = (int)BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;
            EnsureRemaining(frame, offset, size, "payload");

            ReadOnlySpan<byte> payload = frame.Slice(offset, size);
            byte[] decoded = compression == CompressionMethod.Gzip
                ? GzipCodec.Decompress(payload)
                : payload.ToArray();

            return new ServerMessage
            {
                Type = ServerMessageType.FullResponse,
                Sequence = sequence,
                HasSequence = hasSequence,
                IsLastPackage = isLast,
                PayloadJson = Encoding.UTF8.GetString(decoded),
            };
        }

        private static ServerMessage DecodeError(ReadOnlySpan<byte> frame, int offset, CompressionMethod compression)
        {
            EnsureRemaining(frame, offset, 4, "error code");
            uint code = BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;

            EnsureRemaining(frame, offset, 4, "error message size");
            int size = (int)BinaryPrimitives.ReadUInt32BigEndian(frame.Slice(offset, 4));
            offset += 4;
            EnsureRemaining(frame, offset, size, "error message");

            ReadOnlySpan<byte> payload = frame.Slice(offset, size);
            byte[] decoded = compression == CompressionMethod.Gzip
                ? GzipCodec.Decompress(payload)
                : payload.ToArray();

            return new ServerMessage
            {
                Type = ServerMessageType.Error,
                ErrorCode = code,
                ErrorMessage = Encoding.UTF8.GetString(decoded),
            };
        }

        private static void EnsureRemaining(ReadOnlySpan<byte> frame, int offset, int need, string what)
        {
            if (offset + need > frame.Length)
                throw new SeedAsrProtocolException(
                    $"Truncated frame: need {need} bytes for {what} at offset {offset}, have {frame.Length - offset}.");
        }
    }
}
