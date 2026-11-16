using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// Typed near-end port bundle. The composition root holds this value instead of one object cast
/// to seven interfaces, so every consumer sees exactly the facet it owns.
/// </summary>
internal sealed class NearEndPorts
{
    public required INearEndCapturePort Capture { get; init; }

    public required INearEndRenderPort Render { get; init; }

    public required INearEndEndpointPort Endpoint { get; init; }

    public required INearEndConfirmationStore Confirmation { get; init; }

    public required INearEndCaptureControlPort CaptureControl { get; init; }

    public required INearEndResetPort Reset { get; init; }

    public required ISpeechTimelineLocator Timeline { get; init; }

    /// <summary>
    /// Structurally disabled near-end module for compositions without an installed Barge-in
    /// detector. This is a no-op capability, not a runtime fallback: a requested Barge-in module
    /// that lacks a required dependency fails at composition time.
    /// </summary>
    public static NearEndPorts Disabled() => new()
    {
        Capture = NullActiveNearEnd.Instance,
        Render = NullActiveNearEnd.Instance,
        Endpoint = NullActiveNearEnd.Instance,
        Confirmation = NullActiveNearEnd.Instance,
        CaptureControl = NullActiveNearEnd.Instance,
        Reset = NullActiveNearEnd.Instance,
        Timeline = NullActiveNearEnd.Instance,
    };
}
