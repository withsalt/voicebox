namespace VoiceBox.Assistant.Composition.Plans;

/// <summary>
/// Identifies which audio transport the builder resolved. Optional capability modules are recorded
/// as capability-graph facts rather than as separate profiles.
/// </summary>
internal enum AssistantCompositionProfile
{
    /// <summary>A single duplex endpoint provides both capture and playback.</summary>
    BasicDuplexQa,

    /// <summary>Independent record and play endpoints provide capture and playback.</summary>
    BasicSplitAudioQa,
}
