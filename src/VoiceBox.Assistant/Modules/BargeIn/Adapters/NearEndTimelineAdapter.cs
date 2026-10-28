using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Speech-timeline locator facet of the near-end data plane.</summary>
internal sealed class NearEndTimelineAdapter : ISpeechTimelineLocator
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndTimelineAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
        => _dataPlane.TryLocate(audio, out streamEpoch, out startSample, out endSample);
}
