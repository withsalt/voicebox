using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// One composed near-end module: its typed ports plus the state owners. Composition uses only
/// <see cref="Ports"/>; the component references exist so state-machine invariant tests can drive
/// one owner directly without reflection.
/// </summary>
internal sealed class NearEndModule
{
    public required NearEndPorts Ports { get; init; }

    public required NearEndDataPlane DataPlane { get; init; }

    public required NearEndTimelineBuffer Timeline { get; init; }

    public required NearEndEndpointTracker Endpoint { get; init; }

    public required NearEndCandidateStateMachine Candidate { get; init; }

    public required NearEndCaptureStateMachine Capture { get; init; }

    public required NearEndConfirmationJournal Journal { get; init; }

    public required NearEndReplayBuffer Replay { get; init; }

    public required PlaybackDuckingPolicy Ducking { get; init; }
}
