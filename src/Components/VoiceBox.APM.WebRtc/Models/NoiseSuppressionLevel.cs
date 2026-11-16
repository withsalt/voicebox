namespace VoiceBox.APM.WebRtc.Models
{
    /// <summary>
    /// 噪声抑制强度。对应 WebRTC <c>AudioProcessing::Config::NoiseSuppression::Level</c>。
    /// </summary>
    public enum NoiseSuppressionLevel
    {
        Low = 0,
        Moderate = 1,
        High = 2,
        VeryHigh = 3
    }
}
