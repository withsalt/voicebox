namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 写入 16-bit PCM WAV (支持多声道; 输入按 frame-major 排列, 即 [s0c0, s0c1, s1c0, s1c1, ...])。
    /// </summary>
    internal static class WavWriter
    {
        public static void WriteInterleaved(string path, ReadOnlySpan<float> samples, int sampleRate, int channels)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (samples.Length % channels != 0)
                throw new ArgumentException("samples 长度必须能被 channels 整除。", nameof(samples));

            const int bitsPerSample = 16;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int blockAlign = channels * bitsPerSample / 8;
            int dataSize = samples.Length * sizeof(short);

            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);

            writer.Write("RIFF"u8);
            writer.Write(36 + dataSize);
            writer.Write("WAVE"u8);

            writer.Write("fmt "u8);
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)blockAlign);
            writer.Write((short)bitsPerSample);

            writer.Write("data"u8);
            writer.Write(dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                float clamped = Math.Clamp(samples[i], -1f, 1f);
                writer.Write((short)Math.Round(clamped * 32767f));
            }
        }
    }
}
