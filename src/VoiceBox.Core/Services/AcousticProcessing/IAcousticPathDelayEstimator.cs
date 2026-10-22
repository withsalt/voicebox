namespace VoiceBox.Core.Services.AcousticProcessing;

/// <summary>
/// 有状态的在线 render 到 capture 声学路径延迟估计器。
/// </summary>
/// <remarks>
/// 实例由其创建方独占，不得跨音频流共享。<see cref="TryEstimate"/> 必须在单一实时线程调用，
/// 且不得与 <see cref="Reset"/> 或 <see cref="IDisposable.Dispose"/> 并发。
/// </remarks>
public interface IAcousticPathDelayEstimator : IDisposable
{
    /// <summary>
    /// 输入等长、时间对齐且符合创建格式的 render/capture 样本。
    /// </summary>
    /// <returns>
    /// 产生新估计时返回 <see langword="true"/>；仍在预热或本次没有新估计时返回
    /// <see langword="false"/>。实现错误不得伪装成“无估计”，应抛出异常。
    /// </returns>
    bool TryEstimate(
        ReadOnlySpan<float> render,
        ReadOnlySpan<float> capture,
        out AcousticPathDelayEstimate estimate);

    /// <summary>重置流状态并保留格式配置。</summary>
    void Reset();
}
