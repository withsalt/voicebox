namespace VoiceBox.TTS.MossNano.Internal
{
    /// <summary>
    /// 简易 PCM WAV 读取器, 仅供 voice-clone 参考音频使用。
    /// 严格要求输入: 48 kHz 立体声、16-bit PCM (与 MOSS-Audio-Tokenizer-Nano 的 codec_config 完全一致)。
    /// 不引入重采样/格式转换依赖, 把这些预处理交给上游 (FFmpeg 等)。
    /// 返回布局: float32, channel-major, shape = [1, channels, samples], 与 Python `_load_reference_audio` 输出一致。
    /// </summary>
    internal static class ReferenceWavReader
    {
        public const int ExpectedSampleRate = 48000;
        public const int ExpectedChannels = 2;

        public static float[] LoadAsChannelMajor(string path, int expectedSampleRate, int expectedChannels)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            if (!File.Exists(path))
                throw new FileNotFoundException("参考音频文件不存在。", path);

            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            // RIFF header. 本读取器严格要求 little-endian PCM (后续 ReadInt16/ReadInt32 与
            // BitConverter 均按小端解析), RIFX (big-endian) 并不在支持范围内, 不接受其头。
            if (!Match(reader.ReadBytes(4), "RIFF"u8))
                throw new InvalidDataException("不是合法的 RIFF/WAV 文件。");
            _ = reader.ReadUInt32(); // file size - 8
            if (!Match(reader.ReadBytes(4), "WAVE"u8))
                throw new InvalidDataException("RIFF 主类型不是 WAVE。");

            short audioFormat = 0;
            short channels = 0;
            int sampleRate = 0;
            short bitsPerSample = 0;
            byte[]? pcmBytes = null;

            while (stream.Position < stream.Length)
            {
                byte[] chunkId = reader.ReadBytes(4);
                if (chunkId.Length < 4) break;
                int chunkSize = reader.ReadInt32();
                if (chunkSize < 0 || chunkSize > stream.Length - stream.Position)
                    throw new InvalidDataException("WAV 子块大小非法。");

                if (Match(chunkId, "fmt "u8))
                {
                    audioFormat = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    sampleRate = reader.ReadInt32();
                    _ = reader.ReadInt32(); // byteRate
                    _ = reader.ReadInt16(); // blockAlign
                    bitsPerSample = reader.ReadInt16();
                    int consumed = 16;
                    int remaining = chunkSize - consumed;
                    if (remaining > 0) stream.Seek(remaining, SeekOrigin.Current);
                }
                else if (Match(chunkId, "data"u8))
                {
                    pcmBytes = reader.ReadBytes(chunkSize);
                    if (pcmBytes.Length != chunkSize)
                        throw new InvalidDataException("WAV data 块读取不完整。");
                    // data 之后通常就是文件结束; 即便有 LIST/INFO 也无需读取。
                    break;
                }
                else
                {
                    stream.Seek(chunkSize, SeekOrigin.Current);
                }

                if ((chunkSize & 1) == 1 && stream.Position < stream.Length)
                    stream.Seek(1, SeekOrigin.Current); // word alignment padding
            }

            if (pcmBytes is null)
                throw new InvalidDataException("WAV 缺少 data 块。");
            if (audioFormat != 1)
                throw new NotSupportedException($"仅支持 PCM (audioFormat=1), 实际为 {audioFormat}。可使用 ffmpeg 转换。");
            if (bitsPerSample != 16)
                throw new NotSupportedException($"仅支持 16-bit PCM, 实际 {bitsPerSample}-bit。");
            if (sampleRate != expectedSampleRate)
                throw new NotSupportedException(
                    $"参考音频采样率必须为 {expectedSampleRate}Hz, 实际 {sampleRate}Hz。请用 ffmpeg 先重采样。");
            if (channels != expectedChannels)
                throw new NotSupportedException(
                    $"参考音频声道数必须为 {expectedChannels}, 实际 {channels}。请用 ffmpeg 调整 (-ac {expectedChannels})。");

            int totalSamples = pcmBytes.Length / (bitsPerSample / 8);
            int samplesPerChannel = totalSamples / channels;

            // 输出 channel-major, 即 [c, t]: ONNX 期望 [1, channels, samples]。
            float[] channelMajor = new float[1 * channels * samplesPerChannel];
            const float inv32k = 1f / 32768f;
            int srcOffset = 0;
            for (int t = 0; t < samplesPerChannel; t++)
            {
                for (int c = 0; c < channels; c++)
                {
                    short s = BitConverter.ToInt16(pcmBytes, srcOffset);
                    srcOffset += 2;
                    channelMajor[c * samplesPerChannel + t] = s * inv32k;
                }
            }
            return channelMajor;
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
