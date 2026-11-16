namespace VoiceBox.Core.Services.AcousticProcessing;

/// <summary>
/// 为独立音频流创建声学路径延迟估计器。
/// </summary>
/// <remarks>
/// 工厂本身由组合根借用；返回的估计器由调用方拥有并负责释放。
/// </remarks>
public interface IAcousticPathDelayEstimatorFactory
{
    /// <summary>创建采用指定采样率与声道数的全新估计器。</summary>
    IAcousticPathDelayEstimator Create(int sampleRate, int channelCount);
}
