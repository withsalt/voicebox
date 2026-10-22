namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 音频 Xrun 事件参数
    /// </summary>
    public class AudioXrunEventArgs : EventArgs
    {
        /// <summary>
        /// Xrun 类型
        /// </summary>
        public AudioXrunType XrunType { get; }

        /// <summary>
        /// 发生 xrun 时的帧数
        /// </summary>
        public uint FrameCount { get; }

        public AudioXrunEventArgs(AudioXrunType xrunType, uint frameCount)
        {
            XrunType = xrunType;
            FrameCount = frameCount;
        }

        public override string ToString()
        {
            return $"[AudioXrun] Type: {XrunType}, FrameCount: {FrameCount}";
        }
    }
}
