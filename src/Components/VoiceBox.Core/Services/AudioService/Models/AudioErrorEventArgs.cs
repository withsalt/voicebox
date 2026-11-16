namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 音频错误事件参数
    /// </summary>
    public class AudioErrorEventArgs : EventArgs
    {
        /// <summary>
        /// 错误异常
        /// </summary>
        public Exception Exception { get; }

        /// <summary>
        /// 错误消息
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// 错误发生时的状态
        /// </summary>
        public AudioServiceState State { get; }

        public AudioErrorEventArgs(Exception exception, string message, AudioServiceState state)
        {
            Exception = exception ?? throw new ArgumentNullException(nameof(exception));
            Message = message ?? exception.Message;
            State = state;
        }
    }
}
