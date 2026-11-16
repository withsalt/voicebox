using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Lifecycle;

namespace VoiceBox.Assistant.Modules.Recognition.Models;

/// <summary>How a completed recognition result meets the turn's identity gate.</summary>
internal enum RecognitionCompletionGate
{
    /// <summary>No pending identity: project to the Kernel now.</summary>
    Proceed = 0,

    /// <summary>Single-talk verdict still pending: hold the result until AuthorizationResolved.</summary>
    Park = 1,

    /// <summary>Single-talk verdict denied: discard the command and complete the turn without a response.</summary>
    Drop = 2,
}
