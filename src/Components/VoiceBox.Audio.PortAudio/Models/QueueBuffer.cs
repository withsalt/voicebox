using System.Buffers;
using System.Diagnostics;

namespace VoiceBox.Audio.PortAudio.Models
{
    internal sealed class QueueBuffer
    {
        private readonly object _locker = new object();
        private volatile bool _isReleased = false;

        public QueueBuffer(float[] data, int Length)
        {
            this.Data = data ?? throw new ArgumentNullException(nameof(data));
            this.Length = Length;
        }

        public float[] Data { get; }

        public int Length { get; }

        public void Release()
        {
            if (_isReleased)
                return;

            lock (_locker)
            {
                if (_isReleased)
                    return;
                try
                {
                    ArrayPool<float>.Shared.Return(Data, true);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[ERROR] Failed to return QueueBuffer to ArrayPool: {ex.Message}");
                }
                finally
                {
                    _isReleased = true;
                }
            }
        }
    }
}
