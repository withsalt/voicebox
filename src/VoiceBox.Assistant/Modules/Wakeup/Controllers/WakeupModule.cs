using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceBox.Assistant.Modules.Wakeup.Controllers;

internal sealed class WakeupModule : IWakeupModule
{
    private readonly WakeupSessionGate _gate;

    public WakeupModule(
        IWakeupService wakeup,
        int preRollSamples,
        TimeSpan idleTimeout,
        int sampleRate,
        int channels,
        int replayLeadInMs,
        ISpeakerEvidenceProvider? speakerEvidence,
        TimeProvider? timeProvider = null,
        int wakeupStrongScoreMargin = 15,
        int postTriggerEvidenceSamples = 0,
        ILogger? logger = null)
    {
        _gate = new WakeupSessionGate(
            wakeup,
            preRollSamples,
            idleTimeout,
            sampleRate,
            channels,
            replayLeadInMs,
            speakerEvidence,
            timeProvider,
            wakeupStrongScoreMargin,
            postTriggerEvidenceSamples,
            logger);
    }

    public AssistantWakeMode Mode => _gate.Mode;

    public AssistantTimelineRequirement TimelineRequirement =>
        AssistantTimelineRequirement.ProcessedCaptureTimeline;

    public WakeupAuthorizationOutcome LastDeniedOutcome => _gate.LastDeniedOutcome;

    public void Start() => _gate.Start();

    public void Stop() => _gate.Stop();

    public WakeupAudioDecision OnAudio(in ProcessedCaptureFrame frame) => _gate.OnAudio(in frame);

    public WakeupTimeoutDecision CheckIdleTimeout() => _gate.CheckIdleTimeout();

    public WakeupTimeoutDecision ReturnToWakeWord() => _gate.ReturnToWakeWord();

    public void MarkUserActivity() => _gate.MarkUserActivity();

    public void Dispose() => _gate.Dispose();
}
