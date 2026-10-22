namespace VoiceBox.Core.Services.AcousticProcessing;

/// <summary>
/// 一次 render 到 capture 声学路径延迟估计的不可变结果。
/// </summary>
/// <param name="DelaySamples">按估计器配置采样率计量的非负延迟样本数。</param>
/// <param name="IsRefined">是否为已收敛的精估计；否则为粗估计。</param>
public readonly record struct AcousticPathDelayEstimate(
    int DelaySamples,
    bool IsRefined);
