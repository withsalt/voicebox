namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// Float32 [-1,1] PCM 与 16-bit little-endian PCM (pcm_s16le) 之间的转换。
    /// 接口要求音频内部流为 pcm_s16le，故上行前须由 float 采样点转成 int16 字节。
    /// </summary>
    internal static class PcmConverter
    {
        /// <summary>把 float 采样写入目标字节缓冲 (little-endian int16)，返回写入的字节数。</summary>
        public static int WriteInt16LittleEndian(ReadOnlySpan<float> samples, Span<byte> destination)
        {
            if (destination.Length < samples.Length * 2)
                throw new ArgumentException("Destination buffer too small.", nameof(destination));

            for (int i = 0; i < samples.Length; i++)
            {
                short s = FloatToInt16(samples[i]);
                destination[i * 2] = (byte)(s & 0xFF);
                destination[i * 2 + 1] = (byte)((s >> 8) & 0xFF);
            }
            return samples.Length * 2;
        }

        /// <summary>把 float 采样转为新分配的 int16 little-endian 字节数组。</summary>
        public static byte[] ToInt16LittleEndian(ReadOnlySpan<float> samples)
        {
            var bytes = new byte[samples.Length * 2];
            WriteInt16LittleEndian(samples, bytes);
            return bytes;
        }

        private static short FloatToInt16(float sample)
        {
            // 先钳制到 [-1,1]，再线性映射到 int16。用 32767 缩放并四舍五入，避免削波溢出。
            float clamped = sample < -1f ? -1f : (sample > 1f ? 1f : sample);
            int v = (int)MathF.Round(clamped * 32767f);
            if (v > short.MaxValue) v = short.MaxValue;
            else if (v < short.MinValue) v = short.MinValue;
            return (short)v;
        }
    }
}
