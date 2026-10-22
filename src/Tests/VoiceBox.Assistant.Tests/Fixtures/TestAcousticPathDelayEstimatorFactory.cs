using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.Assistant.Tests;

internal sealed class TestAcousticPathDelayEstimatorFactory
    : IAcousticPathDelayEstimatorFactory
{
    public static TestAcousticPathDelayEstimatorFactory Instance { get; } = new();

    private TestAcousticPathDelayEstimatorFactory()
    {
    }

    public IAcousticPathDelayEstimator Create(int sampleRate, int channelCount)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channelCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(channelCount));
        return new NoEstimateAcousticPathDelayEstimator();
    }

    private sealed class NoEstimateAcousticPathDelayEstimator
        : IAcousticPathDelayEstimator
    {
        public bool TryEstimate(
            ReadOnlySpan<float> render,
            ReadOnlySpan<float> capture,
            out AcousticPathDelayEstimate estimate)
        {
            if (render.Length != capture.Length)
                throw new ArgumentException("Render and capture sample counts must match.");
            estimate = default;
            return false;
        }

        public void Reset()
        {
        }

        public void Dispose()
        {
        }
    }
}
