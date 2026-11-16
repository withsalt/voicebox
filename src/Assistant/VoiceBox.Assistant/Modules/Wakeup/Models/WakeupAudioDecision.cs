using VoiceBox.Assistant.Api.Events;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

internal readonly record struct WakeupAudioDecision(
        WakeupAudioRoute Route,
        float[]? ReplayAudio,
        WakeupEventArgs? Wakeup,
        WakeupTriggerKind TriggerKind,
        AssistantWakeMode OldMode,
        AssistantWakeMode NewMode,
        bool ModeChanged);
