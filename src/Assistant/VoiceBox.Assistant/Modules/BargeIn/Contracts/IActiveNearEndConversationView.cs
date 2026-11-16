namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Read-only conversation context required by near-end candidate admission.</summary>
internal interface IActiveNearEndConversationView
{
    bool IsSpeakingWithoutPendingCandidate { get; }

    /// <summary>The turn whose reply a new candidate would interrupt; owner identity for the stop gate.</summary>
    long CurrentTurnId { get; }
}
