namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// Xrun 类型（音频缓冲区溢出/下溢）
    /// </summary>
    public enum AudioXrunType
    {
        /// <summary>
        /// 输入溢出：录音缓冲区满，数据被丢弃
        /// </summary>
        InputOverflow,

        /// <summary>
        /// 输出下溢：播放缓冲区空，产生静音或噪音
        /// </summary>
        OutputUnderflow
    }
}
