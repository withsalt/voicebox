namespace VoiceBox.Audio.Common.Resampling
{
    /// <summary>
    /// 把任意来源格式的 PCM 流式转换为目标采样率/声道数。先按目标声道做混音 (下混/上混)，
    /// 再对每个目标声道用 Concentus/SpeexDSP 做带限重采样，最后交错输出。
    /// 转换按 chunk 流式进行，跨 chunk 保持重采样相位 (源格式不变时块边界无咔哒声)。
    ///
    /// 来源采样率/声道数在每次 <see cref="Convert"/> 时传入 (可来自 TTS chunk 的自描述格式)，
    /// 变化时自动重建重采样器。等采样率时跳过重采样 (仅做声道处理)。
    /// </summary>
    internal sealed class StreamingAudioFormatConverter : IDisposable
    {
        private readonly int _targetSampleRate;
        private readonly int _targetChannels;
        private readonly int _quality;

        private int _sourceSampleRate;
        private int _sourceChannels;
        private StreamingSpeexResampler[]? _resamplers;   // 每个目标声道一个；null 表示等采样率直通
        private float[][] _channelInput = Array.Empty<float[]>();   // 混音后的每声道工作区
        private float[][] _channelOutput = Array.Empty<float[]>();  // 每声道重采样输出 (ref 复用)
        private bool _disposed;

        public int TargetSampleRate => _targetSampleRate;
        public int TargetChannels => _targetChannels;

        public StreamingAudioFormatConverter(
            int targetSampleRate,
            int targetChannels,
            int quality = 10)
        {
            if (targetSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(targetSampleRate));
            if (targetChannels <= 0) throw new ArgumentOutOfRangeException(nameof(targetChannels));
            if (quality is < 0 or > 10) throw new ArgumentOutOfRangeException(nameof(quality), "Quality must be between 0 and 10.");

            _targetSampleRate = targetSampleRate;
            _targetChannels = targetChannels;
            _quality = quality;
        }

        /// <summary>
        /// 转换一段交错 PCM。<paramref name="sourceSampleRate"/>/<paramref name="sourceChannels"/>
        /// 描述 <paramref name="samples"/> 的格式。结果写入 <paramref name="output"/> (按需扩容)，
        /// 返回写入的样本数 (可能为 0)。
        /// </summary>
        public int Convert(ReadOnlySpan<float> samples, int sourceSampleRate, int sourceChannels, ref float[] output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (sourceSampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sourceSampleRate));
            if (sourceChannels <= 0) throw new ArgumentOutOfRangeException(nameof(sourceChannels));

            EnsureFormat(sourceSampleRate, sourceChannels);

            int sourceFrames = samples.Length / sourceChannels;
            if (sourceFrames <= 0) return 0;

            int alignedLength = sourceFrames * sourceChannels;

            // 完全一致 (格式相同且无需裁剪) → 直接拷贝到 output。
            if (_resamplers is null &&
                sourceChannels == _targetChannels &&
                alignedLength == samples.Length)
            {
                EnsureCapacity(ref output, samples.Length);
                samples.CopyTo(output);
                return samples.Length;
            }

            ReadOnlySpan<float> source = samples.Slice(0, alignedLength);
            DownmixToTargetChannels(source, sourceFrames, sourceChannels);

            if (_resamplers is null)
            {
                // 等采样率：仅声道处理。
                return Interleave(_channelInput, sourceFrames, ref output);
            }

            int convertedFrames = int.MaxValue;
            for (int channel = 0; channel < _targetChannels; channel++)
            {
                StreamingSpeexResampler resampler = _resamplers[channel];
                int produced = resampler.Process(_channelInput[channel].AsSpan(0, sourceFrames), ref _channelOutput[channel]);
                convertedFrames = Math.Min(convertedFrames, produced);
            }

            if (convertedFrames is int.MaxValue or <= 0) return 0;
            return Interleave(_channelOutput, convertedFrames, ref output);
        }

        /// <summary>
        /// 流/句结束时冲洗各声道重采样器的群延迟尾部。等采样率时无尾部。
        /// 结果写入 <paramref name="output"/> (按需扩容)，返回写入的样本数。
        /// </summary>
        public int Flush(ref float[] output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_resamplers is null) return 0;

            int convertedFrames = int.MaxValue;
            for (int channel = 0; channel < _targetChannels; channel++)
            {
                int produced = _resamplers[channel].Flush(ref _channelOutput[channel]);
                convertedFrames = Math.Min(convertedFrames, produced);
            }

            if (convertedFrames is int.MaxValue or <= 0) return 0;
            return Interleave(_channelOutput, convertedFrames, ref output);
        }

        /// <summary>复位重采样相位与历史 (开始新独立流前调用)，保留已构建的滤波器表。</summary>
        public void Reset()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_resamplers is null) return;
            foreach (StreamingSpeexResampler resampler in _resamplers)
                resampler.Reset();
        }

        private void EnsureFormat(int sourceSampleRate, int sourceChannels)
        {
            if (_sourceSampleRate == sourceSampleRate && _sourceChannels == sourceChannels && _channelInput.Length == _targetChannels)
                return;

            _sourceSampleRate = sourceSampleRate;
            _sourceChannels = sourceChannels;

            DisposeResamplers();
            _resamplers = sourceSampleRate == _targetSampleRate
                ? null
                : BuildResamplers(sourceSampleRate);

            _channelInput = new float[_targetChannels][];
            _channelOutput = new float[_targetChannels][];
            for (int channel = 0; channel < _targetChannels; channel++)
            {
                _channelInput[channel] = Array.Empty<float>();
                _channelOutput[channel] = Array.Empty<float>();
            }
        }

        private StreamingSpeexResampler[] BuildResamplers(int sourceSampleRate)
        {
            var resamplers = new StreamingSpeexResampler[_targetChannels];
            for (int channel = 0; channel < _targetChannels; channel++)
                resamplers[channel] = new StreamingSpeexResampler(sourceSampleRate, _targetSampleRate, _quality);
            return resamplers;
        }

        /// <summary>把交错的来源帧混到每个目标声道的独立 mono 工作区 <see cref="_channelInput"/>。</summary>
        private void DownmixToTargetChannels(ReadOnlySpan<float> source, int sourceFrames, int sourceChannels)
        {
            for (int channel = 0; channel < _targetChannels; channel++)
            {
                float[] mono = _channelInput[channel];
                if (mono.Length < sourceFrames)
                {
                    mono = new float[sourceFrames];
                    _channelInput[channel] = mono;
                }
                FillTargetChannel(source, sourceFrames, sourceChannels, _targetChannels, channel, mono);
            }
        }

        private static void FillTargetChannel(
            ReadOnlySpan<float> source,
            int sourceFrames,
            int sourceChannels,
            int targetChannels,
            int targetChannel,
            Span<float> destination)
        {
            if (sourceChannels == 1)
            {
                // 单声道源 → 复制到每个目标声道。
                for (int frame = 0; frame < sourceFrames; frame++)
                    destination[frame] = source[frame];
                return;
            }

            if (targetChannels == 1)
            {
                // 多声道源 → 单声道：取平均。
                for (int frame = 0; frame < sourceFrames; frame++)
                {
                    int offset = frame * sourceChannels;
                    float sum = 0;
                    for (int channel = 0; channel < sourceChannels; channel++)
                        sum += source[offset + channel];
                    destination[frame] = sum / sourceChannels;
                }
                return;
            }

            // 多声道 → 多声道：按位映射，目标声道数超过源时夹取最后一个源声道。
            int mappedChannel = Math.Min(targetChannel, sourceChannels - 1);
            for (int frame = 0; frame < sourceFrames; frame++)
                destination[frame] = source[frame * sourceChannels + mappedChannel];
        }

        private int Interleave(float[][] channels, int frames, ref float[] output)
        {
            int total = frames * _targetChannels;
            EnsureCapacity(ref output, total);

            if (_targetChannels == 1)
            {
                // 单声道：直接拷贝，省去交错内层循环。
                Array.Copy(channels[0], output, frames);
                return total;
            }

            for (int frame = 0; frame < frames; frame++)
            {
                int offset = frame * _targetChannels;
                for (int channel = 0; channel < _targetChannels; channel++)
                    output[offset + channel] = channels[channel][frame];
            }
            return total;
        }

        // output 内容每次都被完整覆盖，扩容时无需保留旧值，直接换更大的数组即可。
        private static void EnsureCapacity(ref float[] buf, int needed)
        {
            if (buf.Length >= needed) return;
            int newSize = buf.Length == 0 ? Math.Max(needed, 256) : buf.Length * 2;
            while (newSize < needed) newSize *= 2;
            buf = new float[newSize];
        }

        private void DisposeResamplers()
        {
            if (_resamplers is null) return;
            foreach (StreamingSpeexResampler resampler in _resamplers)
                resampler.Dispose();
            _resamplers = null;
        }

        public void Dispose()
        {
            if (_disposed) return;
            DisposeResamplers();
            _channelInput = Array.Empty<float[]>();
            _channelOutput = Array.Empty<float[]>();
            _disposed = true;
        }
    }
}
