namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

/// <summary>Runtime-owned turn completion, driven by the <c>CompleteTurn</c> effect.</summary>
internal interface ITurnCompletionPort
{
    void CompleteTurn(long turnId);
}
