using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Models;

internal enum SpeechBoundaryObservationKind
{
    None = 0,
    StartListening = 1,
    ConfirmedListening = 2,
    ConfirmedBargeIn = 3,
    BeginRecognition = 4,

    /// <summary>The speaker verdict of the pending single-talk turn, carried as AuthorizationResolved.</summary>
    SingleTalkAuthorization = 5,
}
