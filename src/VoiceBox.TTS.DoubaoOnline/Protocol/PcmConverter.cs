using System.Buffers.Binary;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// 16-bit little-endian PCM (pcm_s16le) 与 Float32 [-1,1] 之间的转换。
    /// 服务端下发的 <c>pcm</c> 音频为 s16le 字节流，需转成 float 采样点供 Core 的 TTS 契约输出。
    /// </summary>
    internal static class PcmConverter
    {
        /// <summary>
        /// 把 s16le 字节流转为 float 采样数组 (值域 [-1,1])。<paramref name="pcm"/> 长度应为偶数；
        /// 若为奇数则忽略末尾游离字节 (半个样本)。
        /// </summary>
        public static float[] Int16LittleEndianToFloat(ReadOnlySpan<byte> pcm)
        {
            int count = pcm.Length / 2;
            if (count == 0) return Array.Empty<float>();

            var samples = new float[count];
            for (int i = 0; i < count; i++)
            {
                short s = BinaryPrimitives.ReadInt16LittleEndian(pcm.Slice(i * 2, 2));
                samples[i] = s / 32768f;
            }
            return samples;
        }
    }
}
