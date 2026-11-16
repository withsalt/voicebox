using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.APM.WebRtc;

/// <summary>为独立音频流创建 WebRTC AEC3 声学路径延迟估计器。</summary>
public sealed class WebRtcAec3DelayEstimatorFactory : IAcousticPathDelayEstimatorFactory
{
    /// <inheritdoc />
    public IAcousticPathDelayEstimator Create(int sampleRate, int channelCount)
        => new WebRtcAec3DelayEstimator(sampleRate, channelCount);
}
