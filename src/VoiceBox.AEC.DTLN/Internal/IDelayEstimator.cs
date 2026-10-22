namespace VoiceBox.AEC.DTLN.Internal
{
    /// <summary>
    /// 在线"扬声器→麦克风"往返延迟估计器抽象。喂入时间对齐的 render/capture,
    /// 在产生可信新估计(超过死区)时返回样本级延迟,用于驱动 <see cref="DynamicDelayLine"/>。
    /// 实现:<see cref="WebRtcAec3DelayAdapter"/>(WebRTC AEC3)。
    /// </summary>
    internal interface IDelayEstimator : IDisposable
    {
        /// <summary>
        /// 喂入一帧时间对齐的 render(远端参考) 与 capture(麦克风原始采集)。
        /// 返回:产生了超过死区的可信新延迟(样本)时返回该值;否则返回 null。
        /// </summary>
        int? FeedFrame(ReadOnlySpan<float> render, ReadOnlySpan<float> capture);

        /// <summary>是否已经产生过至少一次估计。</summary>
        bool HasEstimate { get; }

        /// <summary>最近一次估计的延迟(样本),无估计时为 0。</summary>
        int LastEstimatedDelay { get; }

        /// <summary>重置内部状态。</summary>
        void Reset();
    }
}
