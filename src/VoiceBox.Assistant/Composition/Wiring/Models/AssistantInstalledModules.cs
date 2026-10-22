using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;

namespace VoiceBox.Assistant.Composition.Wiring.Models;

/// <summary>
/// Build-time installed capability modules. Composition instantiates these; the Runtime shell
/// consumes the typed instances and never constructs optional modules from provider fields.
/// </summary>
internal sealed class AssistantInstalledModules
{
    public AssistantInstalledModules(
        ISpeakerModule? speaker,
        IWakeupModule? wakeup,
        IPlaybackBargeInDetector? bargeInDetector,
        IRecognitionProviderAdapter recognition,
        IResponseGenerator responseGenerator)
    {
        Speaker = speaker;
        Wakeup = wakeup;
        BargeInDetector = bargeInDetector;
        Recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        ResponseGenerator = responseGenerator ?? throw new ArgumentNullException(nameof(responseGenerator));
    }

    public ISpeakerModule? Speaker { get; }

    public IWakeupModule? Wakeup { get; }

    public IPlaybackBargeInDetector? BargeInDetector { get; }

    public IRecognitionProviderAdapter Recognition { get; }

    public IResponseGenerator ResponseGenerator { get; }
}
