using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Tracks the confirmed turn's ASR feed and drains its final audio once. The coordinator protects
/// this state with its data-plane lock.
/// </summary>
internal sealed class NearEndCaptureSession
{
    private long _activeTurnId;

    public long ActiveTurnId
    {
        get => _activeTurnId;
        set => _activeTurnId = value;
    }

    /// <summary>
    /// Lock-free read used by the duplex callback to skip frames before taking the data-plane
    /// lock. It preserves the original volatile read on the active-turn field.
    /// </summary>
    public long ReadActiveTurnIdVolatile() => Volatile.Read(ref _activeTurnId);

    public bool DirectFeed { get; set; }

    public long DirectFeedNextSample { get; set; }

    public long DirectFeedSamples { get; set; }

    public int DirectFeedsInFlight { get; set; }

    public bool EndDrainPending { get; private set; }

    public bool EndDrainForced { get; private set; }

    public bool EndFinalizationClaimed { get; private set; }

    public string EndDrainReason { get; private set; } = "none";

    public long EndDrainStreamEpoch { get; private set; }

    public long EndDrainTargetSample { get; private set; }

    public long EndDrainDeadlineSample { get; private set; }

    public TimeSpan EndDrainSpeechDuration { get; private set; }

    public void BeginEndDrain(
        long streamEpoch,
        long endSampleExclusive,
        TimeSpan speechDuration,
        long timeoutSamples)
    {
        EndDrainPending = true;
        EndDrainSpeechDuration = speechDuration;
        EndDrainStreamEpoch = streamEpoch;
        EndDrainTargetSample = endSampleExclusive;
        EndDrainDeadlineSample = SaturatingAdd(EndDrainTargetSample, timeoutSamples);
    }

    public void MarkEndDrainAligned() => EndDrainReason = "aligned";

    public void ForceEndDrain(string reason)
    {
        if (!EndDrainPending || EndFinalizationClaimed)
            return;

        EndDrainForced = true;
        EndDrainReason = reason;
    }

    /// <summary>
    /// Claims the single terminal finalization for this drain. It requires that no direct feed is
    /// still in flight and that the fed watermark reached the target unless the drain was forced.
    /// </summary>
    public bool TryClaimEndFinalization(
        out TimeSpan speechDuration,
        out ActiveEndDrainOutcome outcome)
    {
        speechDuration = TimeSpan.Zero;
        outcome = default;
        if (!EndDrainPending ||
            EndFinalizationClaimed ||
            DirectFeedsInFlight != 0 ||
            (!EndDrainForced && DirectFeedNextSample < EndDrainTargetSample))
        {
            return false;
        }

        EndFinalizationClaimed = true;
        speechDuration = EndDrainSpeechDuration;
        long fedThrough = Math.Min(DirectFeedNextSample, EndDrainTargetSample);
        outcome = new ActiveEndDrainOutcome(
            EndDrainReason,
            EndDrainStreamEpoch,
            EndDrainTargetSample,
            fedThrough,
            Math.Max(0, EndDrainTargetSample - fedThrough));
        return true;
    }

    public void ClearEndDrain()
    {
        EndDrainPending = false;
        EndDrainForced = false;
        EndFinalizationClaimed = false;
        EndDrainReason = "none";
        EndDrainStreamEpoch = 0;
        EndDrainTargetSample = 0;
        EndDrainDeadlineSample = 0;
        EndDrainSpeechDuration = TimeSpan.Zero;
    }

    // Stryker disable equality: right==0 never overflows; left==(Max-right) yields MaxValue for both > and >=.
    public static long SaturatingAdd(long left, long right)
        => right > 0 && left > long.MaxValue - right ? long.MaxValue : left + right;
    // Stryker restore equality
}
