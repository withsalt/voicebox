using System.Runtime.InteropServices;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 最小 PCM WAV 写入器: float32 [-1,1] → 16-bit mono WAV。
    /// 仅用于 <c>--dump-dir</c> 导出 noisy / enhanced / clean 片段供人工试听, 与评测数据链无关。
    /// </summary>
    internal static class WavWriter
    {
        public static void WriteMono16(string path, ReadOnlySpan<float> samples, int sampleRate = WavReader.ExpectedSampleRate)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);

            int dataBytes = samples.Length * sizeof(short);
            using var stream = File.Create(path);
            using var w = new BinaryWriter(stream);

            // RIFF header
            w.Write("RIFF"u8);
            w.Write(36 + dataBytes);      // ChunkSize
            w.Write("WAVE"u8);
            // fmt  chunk
            w.Write("fmt "u8);
            w.Write(16);                  // Subchunk1Size (PCM)
            w.Write((short)1);            // AudioFormat = PCM
            w.Write((short)1);            // NumChannels = mono
            w.Write(sampleRate);          // SampleRate
            w.Write(sampleRate * sizeof(short)); // ByteRate (mono)
            w.Write((short)sizeof(short));       // BlockAlign
            w.Write((short)16);           // BitsPerSample
            // data chunk
            w.Write("data"u8);
            w.Write(dataBytes);

            var pcm = new short[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                float v = samples[i];
                v = v > 1f ? 1f : (v < -1f ? -1f : v);
                pcm[i] = (short)Math.Round(v * 32767f);
            }
            w.Write(MemoryMarshal.AsBytes<short>(pcm));
        }
    }
}
