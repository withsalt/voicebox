namespace VoiceBox.Core.Services.AcousticProcessing;

/// <summary>
/// 可选的声学处理指标源。该观察接口不拥有底层处理器的生命周期。
/// </summary>
public interface IAcousticProcessingMetricsSource
{
    /// <summary>
    /// 尝试读取最近一次 AEC/APM 运行时统计。尚未产生统计或指标源已失效时返回
    /// <see langword="false"/>。实现可以只填充
    /// <see cref="AcousticProcessingMetrics"/> 支持的部分指标。
    /// 此方法必须支持与实时音频处理跨线程调用，并返回独立的不可变值快照。
    /// </summary>
    bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics);
}
