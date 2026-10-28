using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Capture-termination facet of the near-end data plane.</summary>
internal sealed class NearEndCaptureControlAdapter : INearEndCaptureControlPort
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndCaptureControlAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public bool StopCapture(long turnId, bool startCooldown)
        => _dataPlane.StopCapture(turnId, startCooldown);
}
