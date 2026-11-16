namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 音频队列溢出事件参数
    /// </summary>
    public class AudioQueueOverflowEventArgs : EventArgs
    {
        /// <summary>
        /// 被丢弃的样本数
        /// </summary>
        public int DroppedSamples { get; }

        /// <summary>
        /// 当前队列大小
        /// </summary>
        public int CurrentQueueSize { get; }

        /// <summary>
        /// 最大队列大小
        /// </summary>
        public int MaxQueueSize { get; }

        public AudioQueueOverflowEventArgs(int droppedSamples, int currentQueueSize, int maxQueueSize)
        {
            DroppedSamples = droppedSamples;
            CurrentQueueSize = currentQueueSize;
            MaxQueueSize = maxQueueSize;
        }
    }
}
