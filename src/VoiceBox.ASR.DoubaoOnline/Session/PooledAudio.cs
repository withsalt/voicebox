using System.Buffers;

namespace VoiceBox.ASR.DoubaoOnline.Session
{
    /// <summary>
    /// 一段从 <see cref="ArrayPool{Float}"/> 借出的音频块。发送 pump 消费完采样后把租借的缓冲归还池。
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
