using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal enum SpeakerScheduleResult
{
    Queued,
    AlreadyKnown,
    CandidateBusy,
    PrefetchRejected,
    Overloaded,
    Stopped,
}
