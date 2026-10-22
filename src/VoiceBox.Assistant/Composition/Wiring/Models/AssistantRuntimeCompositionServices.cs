using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

namespace VoiceBox.Assistant.Composition.Wiring.Models;

/// <summary>
/// The finished collaborator set the Runtime facade receives. It carries only what the facade
/// itself needs; every other collaborator is reachable solely through the object that owns it.
/// </summary>
internal sealed class AssistantRuntimeCompositionServices
{
    public required AssistantRuntimeLifecycle Lifecycle { get; init; }
    public required AssistantRuntimeHostSurface HostSurface { get; init; }
    public required ISpeechPlaybackControlPort SpeechPlayback { get; init; }
}
