using VoiceBox.Core;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceAssistantDemo;

/// <summary>
/// Typed service bundle used by the demo composition root. The service manager is retained only
/// as the lifetime owner for the provider instances created by the existing model plans.
/// </summary>
internal sealed class AssistantRequiredServices : IDisposable
{
    private readonly IDisposable _lifetime;

    private AssistantRequiredServices(
        IDisposable lifetime,
        IAudioDuplexService audio,
        IVoiceActivityDetectService vad,
        IASRService asr,
        IStreamLLMService llm,
        IStreamTTSService tts,
        IWakeupService wakeup,
        ISpeakerService speaker,
        IAcousticProcessingMetricsSource acousticDiagnostics)
    {
        _lifetime = lifetime;
        Audio = audio;
        Vad = vad;
        Asr = asr;
        Llm = llm;
        Tts = tts;
        Wakeup = wakeup;
        Speaker = speaker;
        AcousticDiagnostics = acousticDiagnostics;
    }

    public IAudioDuplexService Audio { get; }

    public IVoiceActivityDetectService Vad { get; }

    public IASRService Asr { get; }

    public IStreamLLMService Llm { get; }

    public IStreamTTSService Tts { get; }

    public IWakeupService Wakeup { get; }

    public ISpeakerService Speaker { get; }

    public IAcousticProcessingMetricsSource AcousticDiagnostics { get; }

    public static AssistantRequiredServices FromManager(IAudioServiceManager manager)
    {
        ArgumentNullException.ThrowIfNull(manager);

        IAcousticProcessingMetricsSource acousticDiagnostics =
            manager.GetAcousticProcessingMetricsSource();

        return new AssistantRequiredServices(
            manager,
            manager.GetAudioDuplexInstance(),
            manager.GetVoiceActivityDetectInstance(),
            manager.GetOfflineASRServiceInstance(),
            manager.GetStreamLLMServiceInstance(),
            manager.GetStreamTTSServiceInstance(),
            manager.GetWakeupServiceInstance(),
            manager.GetSpeakerServiceInstance(),
            acousticDiagnostics);
    }

    public void Dispose() => _lifetime.Dispose();
}
