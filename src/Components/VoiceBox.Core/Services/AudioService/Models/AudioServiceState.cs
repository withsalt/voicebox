namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 音频服务状态枚举
    /// </summary>
    public enum AudioServiceState
    {
        /// <summary>
        /// 空闲状态，服务未启动
        /// </summary>
        Idle,

        /// <summary>
        /// 初始化中,正在启动服务
        /// </summary>
        Initializing,

        /// <summary>
        /// 活动状态，服务正在运行
        /// </summary>
        Active,

        /// <summary>
        /// 停止中，正在停止服务
        /// </summary>
        Stopping,

        /// <summary>
        /// 错误状态，服务遇到错误
        /// </summary>
        Error
    }
}
