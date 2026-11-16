using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

/// <summary>
/// Commits exactly one Kernel decision. Only the single control-loop consumer may call it; the
/// implementation rejects every other caller.
/// </summary>
internal interface ITurnSemanticCommitPort
{
    KernelCommitResult Commit(TurnSignal signal);
}
