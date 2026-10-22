using VoiceBox.Assistant.Kernel.Effects;

namespace VoiceBox.Assistant.Kernel.Models;

internal sealed record TurnDecision(
    TurnSnapshot Next,
    IReadOnlyList<AssistantEffect> Effects,
    bool Accepted = true);
