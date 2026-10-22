using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;

namespace VoiceBox.Assistant.Composition.Plans;

internal sealed class AssistantRuntimePlan
{
    public AssistantRuntimePlan(
        IAssistantAudioFrontend audio,
        ITurnBoundarySource turnBoundary,
        IRecognitionProviderAdapter recognition,
        IResponseGeneratorFactory responseGeneration,
        ISpeechSynthesisStreamPort speechSynthesis,
        IWakeupModuleFactory? wakeup,
        ISpeakerModuleFactory? speaker,
        IAcousticProcessingDiagnostics? acousticDiagnostics,
        IPlaybackBargeInDetectorFactory? playbackBargeInDetectorFactory)
    {
        Audio = audio ?? throw new ArgumentNullException(nameof(audio));
        TurnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        Recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        ResponseGeneration = responseGeneration ?? throw new ArgumentNullException(nameof(responseGeneration));
        SpeechSynthesis = speechSynthesis ?? throw new ArgumentNullException(nameof(speechSynthesis));
        Wakeup = wakeup;
        Speaker = speaker;
        AcousticDiagnostics = acousticDiagnostics;
        PlaybackBargeInDetectorFactory = playbackBargeInDetectorFactory;
    }

    public IAssistantAudioFrontend Audio { get; }

    public ITurnBoundarySource TurnBoundary { get; }

    public IRecognitionProviderAdapter Recognition { get; }

    public IResponseGeneratorFactory ResponseGeneration { get; }

    public ISpeechSynthesisStreamPort SpeechSynthesis { get; }

    public IWakeupModuleFactory? Wakeup { get; }

    public ISpeakerModuleFactory? Speaker { get; }

    public IAcousticProcessingDiagnostics? AcousticDiagnostics { get; }

    /// <summary>
    /// Concrete render-aware detector selection is a composition decision; the Runtime consumes
    /// only the typed factory port.
    /// </summary>
    public IPlaybackBargeInDetectorFactory? PlaybackBargeInDetectorFactory { get; }

    public IReadOnlyList<ModuleOwnership> Ownership { get; internal set; } =
        Array.AsReadOnly(Array.Empty<ModuleOwnership>());
}
