namespace VoiceBox.Assistant.Kernel.Signals;

internal readonly record struct KernelFailure(
    string Code,
    string Category,
    string Component,
    string MessageKey);
