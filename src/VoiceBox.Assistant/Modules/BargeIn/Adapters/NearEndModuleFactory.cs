using VoiceBox.Assistant.Modules.BargeIn.Buffering;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.BargeIn.State;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// Builds and validates one near-end module. A requested module without its required capability
/// throws here; there is no runtime algorithm fallback and no dual implementation.
/// </summary>
internal static class NearEndModuleFactory
{
    public static NearEndModule Create(NearEndModuleDependencies dependencies)
    {
        ArgumentNullException.ThrowIfNull(dependencies);
        ArgumentNullException.ThrowIfNull(dependencies.Options);
        ArgumentNullException.ThrowIfNull(dependencies.Conversation);
        ArgumentNullException.ThrowIfNull(dependencies.Decisions);
        ArgumentNullException.ThrowIfNull(dependencies.AudioFormat);
        ArgumentNullException.ThrowIfNull(dependencies.Logger);
        ArgumentNullException.ThrowIfNull(dependencies.Runtime);
        ArgumentNullException.ThrowIfNull(dependencies.Playback);
        ArgumentNullException.ThrowIfNull(dependencies.Control);
        ArgumentNullException.ThrowIfNull(dependencies.RecognitionAudio);

        bool identityRequired = dependencies.SpeakerEvidence is not null;
        BargeInRuntimeOptions options = BargeInRuntimeOptions.From(
            dependencies.Options,
            dependencies.AudioFormat,
            identityRequired);
        INearEndAuthorizationPolicy authorization = identityRequired
            ? SpeakerAuthorizationPolicy.Instance
            : AcousticAuthorizationPolicy.Instance;

        var timeline = new NearEndTimelineBuffer(options.ReservoirSamples);
        var endpoint = new NearEndEndpointTracker(timeline);
        var candidate = new NearEndCandidateStateMachine();
        var capture = new NearEndCaptureStateMachine();
        var journal = new NearEndConfirmationJournal();
        var replay = new NearEndReplayBuffer();
        var ducking = new PlaybackDuckingPolicy();
        var effects = new NearEndEffectExecutor(
            dependencies.Playback,
            dependencies.Control,
            dependencies.RecognitionAudio,
            dependencies.SpeakerEvidence);
        var dataPlane = new NearEndDataPlane(
            options,
            dependencies.Conversation,
            dependencies.Decisions,
            dependencies.AudioFormat,
            dependencies.Logger,
            dependencies.Runtime,
            authorization,
            effects,
            dependencies.TimeProvider ?? TimeProvider.System,
            timeline,
            endpoint,
            candidate,
            capture,
            journal,
            replay,
            ducking);

        return new NearEndModule
        {
            Ports = new NearEndPorts
            {
                Capture = new NearEndCaptureAdapter(dataPlane),
                Render = new NearEndRenderAdapter(dataPlane),
                Endpoint = new NearEndEndpointAdapter(dataPlane),
                Confirmation = new NearEndConfirmationAdapter(dataPlane),
                CaptureControl = new NearEndCaptureControlAdapter(dataPlane),
                Reset = new NearEndResetAdapter(dataPlane),
                Timeline = new NearEndTimelineAdapter(dataPlane),
            },
            DataPlane = dataPlane,
            Timeline = timeline,
            Endpoint = endpoint,
            Candidate = candidate,
            Capture = capture,
            Journal = journal,
            Replay = replay,
            Ducking = ducking,
        };
    }
}
