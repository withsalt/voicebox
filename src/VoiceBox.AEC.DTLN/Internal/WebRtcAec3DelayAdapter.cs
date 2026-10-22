using VoiceBox.APM.WebRtc;
using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.AEC.DTLN.Internal
{
    /// <summary>
    /// 把 WebRTC AEC3 提供的 <see cref="IAcousticPathDelayEstimator"/> 适配成 <see cref="IDelayEstimator"/>。
    /// render/capture 整帧透传给原生估计器(原生侧自带 64 样本分块/缓冲),原生返回的全采样级延迟过一道死区后回传,
    /// 驱动 <see cref="DynamicDelayLine"/>。本身不含 FFT/谱/窗——这些都在 AEC3 原生侧完成。
    /// </summary>
    internal sealed class WebRtcAec3DelayAdapter : IDelayEstimator
    {
        // 死区:延迟变化不超过此值不更新延迟线,避免无谓的过渡切换。约 4ms @16k(一个 AEC3 块)。
        private const int DeadbandSamples = 64;

        private readonly IAcousticPathDelayEstimator _estimator;
        private int _lastEstimatedDelay;
        private int _lastReturnedDelay = int.MinValue;
        private bool _hasEstimate;

        public int LastEstimatedDelay => _lastEstimatedDelay;
        public bool HasEstimate => _hasEstimate;

        public WebRtcAec3DelayAdapter(int sampleRate)
        {
            _estimator = new WebRtcAec3DelayEstimatorFactory().Create(
                sampleRate,
                channelCount: 1);
        }

        public int? FeedFrame(ReadOnlySpan<float> render, ReadOnlySpan<float> capture)
        {
            if (!_estimator.TryEstimate(render, capture, out AcousticPathDelayEstimate estimate))
                return null;

            int delay = estimate.DelaySamples;
            _lastEstimatedDelay = delay;
            _hasEstimate = true;

            if (Math.Abs((long)delay - _lastReturnedDelay) <= DeadbandSamples)
                return null;

            _lastReturnedDelay = delay;
            return delay;
        }

        public void Reset()
        {
            _estimator.Reset();
            _lastReturnedDelay = int.MinValue;
            _hasEstimate = false;
            _lastEstimatedDelay = 0;
        }

        public void Dispose() => _estimator.Dispose();
    }
}
