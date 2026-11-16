using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;

namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Models;

internal readonly record struct KernelDecisionObservation(
        TurnSignal Signal,
        TurnSnapshot Previous,
        TurnSnapshot Next,
        IReadOnlyList<AssistantEffect> Effects,
        bool Accepted);
