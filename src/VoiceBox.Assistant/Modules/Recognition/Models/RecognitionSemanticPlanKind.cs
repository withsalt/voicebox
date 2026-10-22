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

internal enum RecognitionSemanticPlanKind
{
    None = 0,
    SessionReady = 1,
    SpeechEnded = 2,
}
