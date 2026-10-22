using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;
using VoiceBox.Assistant.Runtime.Lifecycle;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal enum BargeInSemanticPlanKind
{
    None = 0,
    InterruptionRequested = 1,
    ValidationAccepted = 2,
    ValidationRejected = 3,
    ValidationLateCompletion = 4,
    AuthorizationResolved = 5,
}
