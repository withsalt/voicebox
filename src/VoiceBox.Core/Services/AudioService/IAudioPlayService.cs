using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Core.Services.AudioService
{
    public interface IAudioPlayService : IAudioService
    {
        /// <summary>
        /// 获取当前是否正在播放音频
        /// </summary>
        bool IsPlaying { get; }

        /// <summary>
        /// 获取当前队列中的数据块数量
        /// </summary>
        int CurrentQueueSize { get; }

        /// <summary>
        /// 获取当前服务状态
        /// </summary>
        AudioServiceState State { get; }

        /// <summary>
        /// 播放完成事件（队列为空且播放停止）
        /// </summary>
        event EventHandler? OnPlaybackCompleted;

        /// <summary>
        /// 队列溢出事件（数据被丢弃）
        /// </summary>
        event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;

        /// <summary>
        /// 错误事件
        /// </summary>
        event EventHandler<AudioErrorEventArgs>? OnError;

        /// <summary>
        /// Xrun 事件（音频缓冲区下溢/上溢）
        /// </summary>
        event EventHandler<AudioXrunEventArgs>? OnXrun;

        /// <summary>
        /// 获取默认输出设备索引
        /// </summary>
        int GetDefaultOutputDevice();

        /// <summary>
        /// 异步启动音频播放
        /// </summary>
        /// <param name="deviceIndex">设备索引，-1 表示使用默认设备</param>
        /// <param name="cancellationToken">取消令牌</param>
        Task StartPlayingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default);

        /// <summary>
        /// 异步停止音频播放
        /// </summary>
        /// <param name="cancellationToken">取消令牌</param>
        Task StopPlayingAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// 异步切换播放设备
        /// </summary>
        /// <param name="deviceIndex">新设备索引</param>
        /// <param name="cancellationToken">取消令牌</param>
        Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default);

        /// <summary>
        /// 清空播放队列
        /// </summary>
        void ClearQueue();

        /// <summary>
        /// 尝试向播放队列添加音频数据
        /// </summary>
        /// <param name="data">音频数据</param>
        /// <param name="droppedSamples">如果队列满，返回被丢弃的样本数</param>
        /// <returns>是否成功入队</returns>
        bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples);

        /// <summary>
        /// 设置播放增益 (0..1)，用于 barge-in 期间对当前播放做 ducking（压低音量而非立即清空队列）。
        /// 1 = 原音量。实现可按需做平滑过渡。默认空实现：不支持增益调节的播放器忽略该调用。
        /// </summary>
        /// <param name="gain">目标增益，范围 [0, 1]。</param>
        void SetPlaybackGain(float gain) { }
    }
}
