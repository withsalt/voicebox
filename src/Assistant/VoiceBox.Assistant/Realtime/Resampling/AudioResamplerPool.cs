using VoiceBox.Audio.Common.Resampling.Service;

namespace VoiceBox.Assistant.Realtime.Resampling
{
    /// <summary>
    /// 跨回合缓存一个高保真重采样转换器，避免重复构建滤波表/多相组。
    /// </summary>
    internal sealed class AudioResamplerPool : IDisposable
    {
        private readonly object _lock = new();
        private readonly int _targetSampleRate;
        private readonly int _targetChannels;
        private AudioResampleService? _pooled;
        private bool _pooledInUse;
        private bool _disposed;

        public AudioResamplerPool(int targetSampleRate, int targetChannels)
        {
            if (targetSampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetSampleRate), "Target sample rate must be > 0.");
            if (targetChannels <= 0)
                throw new ArgumentOutOfRangeException(nameof(targetChannels), "Target channels must be > 0.");

            _targetSampleRate = targetSampleRate;
            _targetChannels = targetChannels;
        }

        /// <summary>
        /// 借用跨回合缓存的重采样转换器。正常无重叠时返回缓存实例并 Reset；若缓存实例正被
        /// 另一回合占用，则返回一次性实例，避免并发共享有状态对象。
        /// </summary>
        public Lease Acquire()
        {
            lock (_lock)
            {
                if (!_disposed && !_pooledInUse)
                {
                    _pooled ??= CreateResampler();
                    _pooled.Reset();
                    _pooledInUse = true;
                    return new Lease(this, _pooled, pooled: true);
                }
            }

            return new Lease(this, CreateResampler(), pooled: false);
        }

        public void Dispose()
        {
            AudioResampleService? pooled = null;
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                if (!_pooledInUse)
                {
                    pooled = _pooled;
                    _pooled = null;
                }
            }

            pooled?.Dispose();
        }

        private AudioResampleService CreateResampler()
            => new(new AudioResampleOptions
            {
                TargetSampleRate = _targetSampleRate,
                TargetChannels = _targetChannels,
            });

        private void Release(AudioResampleService resampler, bool pooled)
        {
            if (pooled)
            {
                AudioResampleService? disposeAfterRelease = null;
                lock (_lock)
                {
                    _pooledInUse = false;
                    if (_disposed && ReferenceEquals(_pooled, resampler))
                    {
                        disposeAfterRelease = _pooled;
                        _pooled = null;
                    }
                }

                disposeAfterRelease?.Dispose();
            }
            else
            {
                resampler.Dispose();
            }
        }

        public readonly struct Lease : IDisposable
        {
            private readonly AudioResamplerPool _owner;
            private readonly bool _pooled;

            internal Lease(AudioResamplerPool owner, AudioResampleService resampler, bool pooled)
            {
                _owner = owner;
                Resampler = resampler;
                _pooled = pooled;
            }

            public AudioResampleService Resampler { get; }

            public void Dispose() => _owner.Release(Resampler, _pooled);
        }
    }
}
