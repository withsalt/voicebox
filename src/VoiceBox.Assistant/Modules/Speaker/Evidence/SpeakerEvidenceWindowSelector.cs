using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;

namespace VoiceBox.Assistant.Modules.Speaker.Evidence;

/// <summary>
/// Pure window geometry for speaker evidence on the identity timeline. Windows are hop-aligned to
/// the current contiguous segment origin, so two callers asking about the same speech always name
/// the same cache keys.
/// </summary>
internal sealed class SpeakerEvidenceWindowSelector
{
    private readonly SpeakerVerificationPolicy _policy;

    public SpeakerEvidenceWindowSelector(SpeakerVerificationPolicy policy)
        => _policy = policy ?? throw new ArgumentNullException(nameof(policy));

    public int WindowSamples => _policy.WindowSamples;

    public int HopSamples => _policy.HopSamples;

    /// <summary>
    /// Computes the hop-aligned geometry available to a wakeup candidate. The verifier ranks the
    /// complete windows by speech coverage; this method supplies the candidate/required-end
    /// metadata used while that ranking is still waiting for audio to arrive.
    /// </summary>
    public EvidenceWindowSelection SelectWakeupWindows(
        in SpeakerEvidenceQuery query,
        int windowCount,
        long segmentOrigin,
        long retainedEnd)
    {
        long desiredStart = query.CandidateStartSample;
        if (query.HasFocusInterval)
        {
            long coverage = checked(_policy.WindowSamples + (long)(windowCount - 1) * _policy.HopSamples);
            long focusCenter = checked(query.FocusStartSample +
                (query.FocusEndSampleExclusive - query.FocusStartSample) / 2);
            desiredStart = Math.Max(focusCenter - coverage / 2, query.CandidateStartSample);
        }

        long firstWindowStart = AlignUp(desiredStart, segmentOrigin, _policy.HopSamples);
        long firstWindowEnd = checked(firstWindowStart + _policy.WindowSamples);
        long requiredEnd = checked(firstWindowEnd + (long)(windowCount - 1) * _policy.HopSamples);
        long availableEnd = Math.Min(query.CandidateEndSampleExclusive, retainedEnd);
        int availableCount = ComputeAvailableWindowCount(firstWindowStart, availableEnd, windowCount);
        int candidateCount = ComputeAvailableWindowCount(firstWindowStart, query.CandidateEndSampleExclusive, windowCount);
        return new EvidenceWindowSelection(
            firstWindowStart,
            availableCount,
            candidateCount,
            availableCount,
            requiredEnd);
    }

    /// <summary>The most recent complete windows inside the candidate; the selection slides as audio arrives.</summary>
    public EvidenceWindowSelection SelectRecentWindows(
        in SpeakerEvidenceQuery query,
        int windowCount,
        long segmentOrigin,
        long retainedEnd)
    {
        long earliestWindowStart = AlignUp(query.CandidateStartSample, segmentOrigin, _policy.HopSamples);
        long earliestWindowEnd = checked(earliestWindowStart + _policy.WindowSamples);
        long requiredEnd = checked(earliestWindowEnd + (long)(windowCount - 1) * _policy.HopSamples);
        int candidateCount = ComputeAvailableWindowCount(
            earliestWindowStart,
            query.CandidateEndSampleExclusive,
            int.MaxValue);
        int retainedCount = ComputeAvailableWindowCount(
            earliestWindowStart,
            Math.Min(query.CandidateEndSampleExclusive, retainedEnd),
            int.MaxValue);
        int selectedCount = Math.Min(windowCount, retainedCount);
        long firstSelectedStart = selectedCount == 0
            ? earliestWindowStart
            : checked(earliestWindowStart + (long)(retainedCount - selectedCount) * _policy.HopSamples);
        return new EvidenceWindowSelection(
            firstSelectedStart,
            selectedCount,
            candidateCount,
            retainedCount,
            requiredEnd);
    }

    /// <summary>All complete hop-aligned windows whose end lies in <c>(afterEnd, retainedEnd]</c>, newest last.</summary>
    public int EnumerateCompleteWindowEnds(
        long segmentOrigin,
        long afterEnd,
        long retainedEnd,
        Span<long> destination)
    {
        long firstEnd = checked(segmentOrigin + _policy.WindowSamples);
        if (retainedEnd < firstEnd)
            return 0;
        long lastEnd = firstEnd + (retainedEnd - firstEnd) / _policy.HopSamples * _policy.HopSamples;
        int count = 0;
        for (long end = lastEnd; end > afterEnd && end >= firstEnd && count < destination.Length; end -= _policy.HopSamples)
            destination[count++] = end;
        destination.Slice(0, count).Reverse();
        return count;
    }

    private int ComputeAvailableWindowCount(long firstWindowStart, long availableEnd, int maximum)
    {
        long firstWindowEnd = checked(firstWindowStart + _policy.WindowSamples);
        if (availableEnd < firstWindowEnd)
            return 0;

        long count = (availableEnd - firstWindowEnd) / _policy.HopSamples + 1;
        return checked((int)Math.Min(count, maximum));
    }

    internal static long AlignUp(long value, long origin, int step)
    {
        long relative = checked(value - origin);
        if (relative <= 0)
            return origin;
        long remainder = relative % step;
        return remainder == 0 ? value : checked(value + step - remainder);
    }
}
