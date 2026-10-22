using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;

internal interface IRenderDelayEstimator : IDisposable
{
    RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone);

    void Reset();
}
