using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal readonly record struct SpeakerVerificationDiagnosticsSnapshot(
    long Queries,
    long Matches,
    long Pending,
    long Mismatches,
    long InsufficientEvidence,
    long Unavailable,
    long Cancelled,
    long CacheHits,
    long CacheMisses,
    long InferenceScheduled,
    long PrefetchScheduled,
    long PrefetchRejected,
    long InferenceOverload,
    long InferenceCompleted,
    long InferenceLatencyTicks,
    long InferenceLatencyMaxTicks,
    long TimelineGaps,
    long AudioExpired,
    long InvalidFrames,
    long AppendedFrames,
    long MaxQueueDepth,
    double QueueWaitMilliseconds,
    long MissingSamples,
    long QualityRejected,
    long FinalCandidates)
{
    public double AverageInferenceMilliseconds
        => InferenceCompleted == 0
            ? 0
            : InferenceLatencyTicks * 1000d / System.Diagnostics.Stopwatch.Frequency / InferenceCompleted;

    public double MaxInferenceMilliseconds
        => InferenceLatencyMaxTicks <= 0
            ? 0
            : InferenceLatencyMaxTicks * 1000d / System.Diagnostics.Stopwatch.Frequency;
}
