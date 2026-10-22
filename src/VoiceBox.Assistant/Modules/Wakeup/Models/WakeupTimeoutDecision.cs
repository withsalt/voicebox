using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

/// <summary>
/// Wake-mode outcome returned by <see cref="IWakeupModule"/>. It lives beside the port so callers
/// can consume the decision without referencing the wakeup module's concrete types.
/// </summary>
internal readonly record struct WakeupTimeoutDecision(
        AssistantWakeMode OldMode,
        AssistantWakeMode NewMode,
        bool ModeChanged);
