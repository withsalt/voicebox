namespace VoiceBox.Audio.Common.Resampling.Service
{
    /// <summary>
    /// <see cref="IAudioResampleService"/> 的默认实现：包一个流式 <see cref="StreamingAudioFormatConverter"/>，
    /// 用 Concentus/SpeexDSP 做高保真采样率/声道转换。有状态、单流使用。
    /// </summary>
    public sealed class AudioResampleService : IAudioResampleService
    {
        private readonly AudioResampleOptions _options;
        private readonly StreamingAudioFormatConverter _converter;
        private readonly Lock _lock = new();
        private bool _disposed;

        public int TargetSampleRate => _options.TargetSampleRate;
        public int TargetChannels => _options.TargetChannels;

        public AudioResampleService(AudioResampleOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));

            if (_options.TargetSampleRate <= 0)
                throw new ArgumentException("TargetSampleRate must be > 0.", nameof(options));
            if (_options.TargetChannels <= 0)
                throw new ArgumentException("TargetChannels must be > 0.", nameof(options));
            if (_options.Quality is < 0 or > 10)
                throw new ArgumentException("Quality must be between 0 and 10.", nameof(options));
            if (_options.SourceSampleRate is <= 0)
                throw new ArgumentException("SourceSampleRate must be null or > 0.", nameof(options));
            if (_options.SourceChannels is <= 0)
                throw new ArgumentException("SourceChannels must be null or > 0.", nameof(options));

            _converter = new StreamingAudioFormatConverter(
                _options.TargetSampleRate,
                _options.TargetChannels,
                _options.Quality);
        }

        public int Convert(ReadOnlySpan<float> samples, int sourceSampleRate, int sourceChannels, ref float[] output)
        {
            int rate = ResolveSourceRate(sourceSampleRate);
            int channels = ResolveSourceChannels(sourceChannels);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return _converter.Convert(samples, rate, channels, ref output);
            }
        }

        public int Flush(ref float[] output)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return _converter.Flush(ref output);
            }
        }

        public float[] Convert(ReadOnlySpan<float> samples, int sourceSampleRate, int sourceChannels)
        {
            float[] buffer = Array.Empty<float>();
            int n = Convert(samples, sourceSampleRate, sourceChannels, ref buffer);
            return ToExactArray(buffer, n);
        }

        public float[] Convert(ReadOnlySpan<float> samples) => Convert(samples, 0, 0);

        public float[] Flush()
        {
            float[] buffer = Array.Empty<float>();
            int n = Flush(ref buffer);
            return ToExactArray(buffer, n);
        }

        private int ResolveSourceRate(int sourceSampleRate) => sourceSampleRate > 0
            ? sourceSampleRate
            : _options.SourceSampleRate
                ?? throw new InvalidOperationException(
                    "无法确定来源采样率：Convert 未传入有效值且 AudioResampleOptions.SourceSampleRate 未配置。");

        private int ResolveSourceChannels(int sourceChannels) => sourceChannels > 0
            ? sourceChannels
            : _options.SourceChannels
                ?? throw new InvalidOperationException(
                    "无法确定来源声道数：Convert 未传入有效值且 AudioResampleOptions.SourceChannels 未配置。");

        private static float[] ToExactArray(float[] buffer, int n)
            => n == 0 ? Array.Empty<float>()
             : n == buffer.Length ? buffer
             : buffer.AsSpan(0, n).ToArray();

        public void Reset()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                _converter.Reset();
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _converter.Dispose();
                _disposed = true;
            }
        }
    }
}
