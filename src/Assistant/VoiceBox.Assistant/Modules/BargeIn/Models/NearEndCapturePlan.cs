namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Immutable plan for the capture that follows an authorized interruption. It is built at
/// authorization time from the frozen decision and the known sentence start, and may be committed
/// before the recognition timeline has reached that start. <see cref="Ready"/> means the lease also
/// carries an immediate pre-roll snapshot; a pending plan is activated once the main timeline
/// reaches <see cref="StartSample"/>.
/// </summary>
internal readonly record struct NearEndCapturePlan(
    long StreamEpoch,
    long StartSample,
    string Source,
    bool EndpointVadCandidate,
    bool PreferEndpointVadStart,
    bool Ready);
