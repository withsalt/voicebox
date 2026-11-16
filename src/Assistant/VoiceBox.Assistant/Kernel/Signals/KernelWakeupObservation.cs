using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Kernel.Signals;

internal readonly record struct KernelWakeupObservation(
    AssistantWakeMode FromMode,
    AssistantWakeMode ToMode,
    bool ModeChanged,
    bool HasWakeup,
    string Keyword,
    int Score,
    int Threshold,
    long DurationMs,
    long StartTimeMs,
    KernelWakeupTriggerKind TriggerKind);
