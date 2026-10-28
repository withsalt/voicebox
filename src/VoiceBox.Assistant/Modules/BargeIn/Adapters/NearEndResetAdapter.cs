using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Lifecycle/reset facet of the near-end data plane.</summary>
internal sealed class NearEndResetAdapter : INearEndResetPort
{
    private readonly NearEndDataPlane _dataPlane;

    public NearEndResetAdapter(NearEndDataPlane dataPlane)
        => _dataPlane = dataPlane ?? throw new ArgumentNullException(nameof(dataPlane));

    public void Reset() => _dataPlane.Reset();

    public void ClearAudioReservoirs() => _dataPlane.ClearAudioReservoirs();
}
