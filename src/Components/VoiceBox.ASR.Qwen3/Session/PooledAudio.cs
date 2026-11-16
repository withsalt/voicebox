using System.Buffers;

namespace VoiceBox.ASR.Qwen3.Session
{
    /// <summary>
    /// One audio chunk borrowed from <see cref="ArrayPool{Float}"/>. The pump returns
    /// the rented buffer to the pool when it is done consuming the samples.
    /// </summary>
    internal readonly struct PooledAudio : IDisposable
    {
        private readonly float[]? _buffer;
        public int Length { get; }

        public PooledAudio(float[] buffer, int length)
        {
            _buffer = buffer;
            Length = length;
        }

        public ReadOnlySpan<float> Span => _buffer is null ? ReadOnlySpan<float>.Empty : _buffer.AsSpan(0, Length);

        public void Dispose()
        {
            if (_buffer is not null)
                ArrayPool<float>.Shared.Return(_buffer);
        }
    }
}
