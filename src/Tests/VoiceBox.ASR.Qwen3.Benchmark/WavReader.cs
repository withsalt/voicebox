using System.Runtime.InteropServices;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 最小 PCM WAV 读取器, 专为 AIShell-1 16kHz / 16-bit / mono 设计。
    /// 输出 float32 [-1, 1], 与 IStreamASRService.FeedAudio 期望一致。
    /// </summary>
    internal static class WavReader
    {
        public const int ExpectedSampleRate = 16000;

        /// <summary>
        /// 读取整个 WAV, 返回单通道 float32 样本。若文件不是 16kHz mono 16-bit 直接抛错,
        /// 让基准测试在采样率/格式异常的样本上立即失败而不是默默给出错误结果。
        /// </summary>
        public static float[] ReadMonoFloat32(string path, out int sampleRate, out int channels)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            if (!File.Exists(path))
                throw new FileNotFoundException("Audio file not found.", path);

            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            if (!Match(reader.ReadBytes(4), "RIFF"u8))
                throw new InvalidDataException($"Not a RIFF/WAV file: {path}");
            _ = reader.ReadUInt32(); // file size
            if (!Match(reader.ReadBytes(4), "WAVE"u8))
                throw new InvalidDataException($"RIFF main type is not WAVE: {path}");

            short audioFormat = 0;
            short bitsPerSample = 0;
            sampleRate = 0;
            channels = 0;
            byte[]? pcmBytes = null;

            while (stream.Position < stream.Length)
            {
                byte[] chunkId = reader.ReadBytes(4);
                if (chunkId.Length < 4) break;
                int chunkSize = reader.ReadInt32();
                if (chunkSize < 0 || chunkSize > stream.Length - stream.Position)
                    throw new InvalidDataException($"WAV sub-chunk size invalid in {path}.");

                if (Match(chunkId, "fmt "u8))
                {
                    audioFormat = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    sampleRate = reader.ReadInt32();
                    _ = reader.ReadInt32(); // byte rate
                    _ = reader.ReadInt16(); // block align
                    bitsPerSample = reader.ReadInt16();
                    int remaining = chunkSize - 16;
                    if (remaining > 0) stream.Seek(remaining, SeekOrigin.Current);
                }
                else if (Match(chunkId, "data"u8))
                {
                    pcmBytes = reader.ReadBytes(chunkSize);
                    if (pcmBytes.Length != chunkSize)
                        throw new InvalidDataException($"WAV data chunk incomplete in {path}.");
                    break;
                }
                else
                {
                    stream.Seek(chunkSize, SeekOrigin.Current);
                }

                if ((chunkSize & 1) == 1 && stream.Position < stream.Length)
                    stream.Seek(1, SeekOrigin.Current);
            }

            if (pcmBytes is null)
                throw new InvalidDataException($"WAV missing data chunk: {path}");
            if (audioFormat != 1)
                throw new NotSupportedException($"Only PCM (audioFormat=1) supported, got {audioFormat} in {path}.");
            if (bitsPerSample != 16)
                throw new NotSupportedException($"Only 16-bit PCM supported, got {bitsPerSample}-bit in {path}.");
            if (channels != 1)
                throw new NotSupportedException($"Only mono supported, got {channels}-channel in {path}.");
            if (sampleRate != ExpectedSampleRate)
                throw new NotSupportedException($"Expected {ExpectedSampleRate}Hz, got {sampleRate}Hz in {path}.");

            var int16 = MemoryMarshal.Cast<byte, short>(pcmBytes.AsSpan());
            float[] result = new float[int16.Length];
            const float inv = 1f / 32768f;
            for (int i = 0; i < int16.Length; i++)
                result[i] = int16[i] * inv;
            return result;
        }

        private static bool Match(ReadOnlySpan<byte> actual, ReadOnlySpan<byte> expected)
        {
            if (actual.Length != expected.Length) return false;
            for (int i = 0; i < actual.Length; i++)
                if (actual[i] != expected[i]) return false;
            return true;
        }
    }
}
