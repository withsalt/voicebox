using System.Buffers;

namespace VoiceBox.ASR.FireRedASR2.Session
{
    internal readonly struct PooledAudio(float[] buffer, int length) : IDisposable
    {
        public readonly float[] Buffer = buffer;
        public readonly int Length = length;

        public ReadOnlySpan<float> Span => Buffer.AsSpan(0, Length);

        public ReadOnlyMemory<float> Memory => Buffer.AsMemory(0, Length);

        public void Dispose()
        {
            if (Buffer is not null)
                ArrayPool<float>.Shared.Return(Buffer);
        }
    }
}
