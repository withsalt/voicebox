using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Contracts;

/// <summary>
/// Optional wakeup module boundary. The gate owns mode, pre-roll, authorization and detector
/// ordering; Runtime receives only typed timeline decisions from this port.
/// </summary>
internal interface IWakeupModule : IDisposable
{
    AssistantTimelineRequirement TimelineRequirement { get; }

    AssistantWakeMode Mode { get; }

    void Start();

    void Stop();

    WakeupAudioDecision OnAudio(in ProcessedCaptureFrame frame);

    WakeupTimeoutDecision CheckIdleTimeout();

    WakeupTimeoutDecision ReturnToWakeWord();

    void MarkUserActivity();
}
