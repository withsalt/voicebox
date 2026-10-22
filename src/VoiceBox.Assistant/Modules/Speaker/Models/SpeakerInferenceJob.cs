using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal readonly record struct SpeakerInferenceJob(
    long StreamEpoch,
    long StartSample,
    long EndSampleExclusive,
    SpeakerInferencePriority Priority,
    SpeakerCandidateId CandidateId,
    bool IsPrefetch,
    bool IsPrefix = false);
