namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Idle listening-threshold and barge-in candidate admission. SpeechCandidateOwner
/// owns the lock and the detector's PCM accumulation; this type only decides whether a
/// speech-start fact begins a candidate or a Listening turn.
/// </summary>
internal enum SpeechCandidateKind
{
    None,
    BargeIn,
    Listening,
}
