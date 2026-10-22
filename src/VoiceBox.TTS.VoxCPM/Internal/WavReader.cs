using System.Buffers.Binary;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// 最小 RIFF/WAVE 读取器 (克隆 prompt 音频用): PCM 16/24/32-bit 与 IEEE float32,
    /// 多声道取均值。只解析标准 fmt/data chunk。
    /// </summary>
    internal static class WavReader
    {
        public static (float[] samples, int sampleRate) ReadMono(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var span = bytes.AsSpan();
            if (span.Length < 44 ||
                !span[..4].SequenceEqual("RIFF"u8) || !span.Slice(8, 4).SequenceEqual("WAVE"u8))
                throw new InvalidDataException($"不是 RIFF/WAVE 文件: {path}");

            int fmtTag = 0, channels = 0, sampleRate = 0, bits = 0;
            int dataOfs = -1, dataLen = 0;
            int pos = 12;
            while (pos + 8 <= span.Length)
            {
                var id = span.Slice(pos, 4);
                int size = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(pos + 4, 4));
                int body = pos + 8;
                if (id.SequenceEqual("fmt "u8))
                {
                    fmtTag = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(body, 2));
                    channels = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(body + 2, 2));
                    sampleRate = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(body + 4, 4));
                    bits = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(body + 14, 2));
                    if (fmtTag == -2 /* WAVE_FORMAT_EXTENSIBLE */ && size >= 40)
                        fmtTag = BinaryPrimitives.ReadInt16LittleEndian(span.Slice(body + 24, 2));
                }
                else if (id.SequenceEqual("data"u8))
                {
                    dataOfs = body;
                    dataLen = Math.Min(size, span.Length - body);
                }
                pos = body + size + (size & 1);   // chunk 按 2 字节对齐
            }
            if (channels <= 0 || sampleRate <= 0 || dataOfs < 0)
                throw new InvalidDataException($"WAVE 缺少 fmt/data chunk: {path}");

            var data = span.Slice(dataOfs, dataLen);
            int bytesPerSample = bits / 8;
            int frames = dataLen / (bytesPerSample * channels);
            var mono = new float[frames];

            for (int f = 0; f < frames; f++)
            {
                float acc = 0f;
                for (int c = 0; c < channels; c++)
                {
                    int o = (f * channels + c) * bytesPerSample;
                    acc += (fmtTag, bits) switch
                    {
                        (1, 16) => BinaryPrimitives.ReadInt16LittleEndian(data.Slice(o, 2)) / 32768f,
                        (1, 24) => (((data[o + 2] << 24) | (data[o + 1] << 16) | (data[o] << 8)) >> 8) / 8388608f,
                        (1, 32) => BinaryPrimitives.ReadInt32LittleEndian(data.Slice(o, 4)) / 2147483648f,
                        (3, 32) => BinaryPrimitives.ReadSingleLittleEndian(data.Slice(o, 4)),
                        _ => throw new NotSupportedException($"不支持的 WAVE 格式: fmt={fmtTag}, bits={bits} ({path})"),
                    };
                }
                mono[f] = acc / channels;
            }
            return (mono, sampleRate);
        }
    }
}
