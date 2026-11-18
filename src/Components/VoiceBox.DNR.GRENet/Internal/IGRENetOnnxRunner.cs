namespace VoiceBox.DNR.GRENet.Internal;

internal interface IGRENetOnnxRunner : IDisposable
{
    int FrequencyBins { get; }

    void Run(ReadOnlySpan<float> amplitude, ReadOnlySpan<float> phase, Span<float> outputAmplitude, Span<float> outputPhase);

    void Reset();
}
