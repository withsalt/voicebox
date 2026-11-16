using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

/// <summary>Scheduling priority; higher runs first. Prefetch always yields to a formal candidate.</summary>
internal enum SpeakerInferencePriority
{
    WakeupPrefetch = 0,
    BargeInPrefetch = 1,
    SingleTalk = 2,
    WakeupCandidate = 3,
    BargeInCommandCandidate = 4,
}
