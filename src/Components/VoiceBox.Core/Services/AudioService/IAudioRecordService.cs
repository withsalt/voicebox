using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Core.Services.AudioService
{
    public interface IAudioRecordService : IAudioService
    {
        /// <summary>
        /// 获取当前是否正在录音
        /// </summary>
        bool IsRecording { get; }

        /// <summary>
        /// 获取当前服务状态
        /// </summary>
        AudioServiceState State { get; }

        /// <summary>
        /// 数据到达事件
        /// </summary>
        event Action<ReadOnlySpan<float>>? OnDataAvailable;

        /// <summary>
        /// 错误事件
        /// </summary>
        event EventHandler<AudioErrorEventArgs>? OnError;

        /// <summary>
        /// Xrun 事件（音频缓冲区下溢/上溢）
        /// </summary>
        event EventHandler<AudioXrunEventArgs>? OnXrun;

        /// <summary>
        /// 获取默认输入设备索引
        /// </summary>
        int GetDefaultInputDevice();

        /// <summary>
        /// 异步开始录音
        /// </summary>
        /// <param name="deviceIndex">要使用的设备索引，-1 表示使用默认设备</param>
        /// <param name="cancellationToken">取消令牌</param>
        Task StartRecordingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default);

        /// <summary>
        /// 异步停止录音
        /// </summary>
        /// <param name="cancellationToken">取消令牌</param>
        Task StopRecordingAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// 异步切换录音设备
        /// </summary>
        /// <param name="deviceIndex">新设备索引</param>
        /// <param name="cancellationToken">取消令牌</param>
        Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default);
    }
}
