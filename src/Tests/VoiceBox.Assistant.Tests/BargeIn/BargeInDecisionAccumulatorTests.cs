using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Window aggregation behaviour. All assertions are expressed in milliseconds and
/// dimensionless fractions so they stay valid regardless of capture gain or device.
/// </summary>
public sealed class BargeInDecisionAccumulatorTests
{
    private const double FrameMs = 20.0;
    private const double WindowMs = 360.0;
    private const double RequiredMs = 180.0;

    private static NearEndFrameDecision Frame(
        bool accepted,
        bool echoLike = false,
        double rejectDecay = 0.35,
        double captureRms = 0.2) =>
        new(
            TimestampTicks: 0L,
            DurationMs: FrameMs,
            Accepted: accepted,
            EchoLike: echoLike,
            RenderActive: true,
            CaptureRms: captureRms,
            RequiredRms: 0.0,
            RejectDecay: rejectDecay,
            Reason: accepted ? "accepted" : "rejected");

    private static BargeInDecision Aggregate(params NearEndFrameDecision[] frames)
    {
        var accumulator = new BargeInDecisionAccumulator(64);
        foreach (NearEndFrameDecision frame in frames)
            accumulator.Add(frame);
        return accumulator.AggregateRecent(
            WindowMs,
            RequiredMs,
            minRms: 0f,
            BargeInDecisionParameters.Default);
    }

    private static NearEndFrameDecision[] Pattern(string pattern, double rejectDecay = 0.35)
    {
        var frames = new NearEndFrameDecision[pattern.Length];
        for (int i = 0; i < pattern.Length; i++)
            frames[i] = Frame(pattern[i] == 'A', rejectDecay: rejectDecay);
        return frames;
    }

    [Fact]
    public void UnbrokenAcceptedRun_Confirms()
    {
        // 9 x 20ms = exactly the required duration.
        BargeInDecision decision = Aggregate(Pattern("AAAAAAAAA"));

        Assert.Equal(BargeInAction.Interrupt, decision.Action);
        Assert.Equal("confirmed-nearend-window", decision.Reason);
    }

    [Fact]
    public void IntraWordDips_DoNotEraseEarnedDuration()
    {
        // Real speech at a 20ms hop dips below the model threshold between syllables. With
        // full decay on evidence-absent frames this pattern never confirms, because each dip
        // cancels a preceding accepted frame. Partial decay must let it through.
        BargeInDecision decision = Aggregate(Pattern("AAAAARAAAAARAAAAA"));

        Assert.Equal(BargeInAction.Interrupt, decision.Action);
    }

    [Fact]
    public void CausalEchoFrames_StillEraseEarnedDurationInFull()
    {
        // Echo-like frames carry positive causal evidence and keep full decay, so an
        // alternating echo pattern must not accumulate confirmation.
        var frames = new List<NearEndFrameDecision>();
        for (int i = 0; i < 18; i++)
            frames.Add(Frame(i % 2 == 0, echoLike: i % 2 != 0, rejectDecay: 1.0));

        BargeInDecision decision = Aggregate(frames.ToArray());

        Assert.Equal(BargeInAction.Continue, decision.Action);
    }

    [Fact]
    public void ScatteredSingleFrames_DoNotConfirm()
    {
        // Partial decay must not let a fragmentary alternating pattern drift to
        // confirmation: no contiguous run reaches the minimum burst length.
        BargeInDecision decision = Aggregate(Pattern("ARARARARARARARARAR"));

        Assert.Equal(BargeInAction.Continue, decision.Action);
    }

    [Fact]
    public void FragmentedEvidenceWithoutContiguousBurst_IsRejectedAsNonContiguous()
    {
        // Two-frame bursts never form a syllable-length run, which is the pattern detector
        // false positives take on far-end single talk.
        BargeInDecision decision = Aggregate(Pattern("AARAARAARAARAARAA"));

        Assert.Equal(BargeInAction.Continue, decision.Action);
        Assert.Equal("insufficient-contiguous-nearend-evidence", decision.Reason);
    }

    [Fact]
    public void TrailingRejection_BlocksConfirmation()
    {
        // The confirmation tail requirement is unchanged: the newest frame must be accepted,
        // so a burst that has already ended does not fire late.
        BargeInDecision decision = Aggregate(Pattern("AAAAAAAAARRRRR", rejectDecay: 1.0));

        Assert.Equal(BargeInAction.Continue, decision.Action);
    }

    [Fact]
    public void ContiguousRunRequirement_TracksRequiredDuration()
    {
        // The minimum burst is a fraction of the required duration rather than a fixed
        // millisecond count, so it follows MinDurationMs instead of assuming one value. The
        // pattern ends on an accepted frame so the confirmation tail rule, which is
        // independent of this fraction, does not mask the effect being measured.
        var accumulator = new BargeInDecisionAccumulator(64);
        foreach (NearEndFrameDecision frame in Pattern("AARAARAARAARAARAA"))
            accumulator.Add(frame);

        BargeInDecision strict = accumulator.AggregateRecent(
            WindowMs,
            RequiredMs,
            minRms: 0f,
            BargeInDecisionParameters.Default);

        Assert.Equal(BargeInAction.Continue, strict.Action);
        Assert.Equal("insufficient-contiguous-nearend-evidence", strict.Reason);

        // At a 60ms requirement the burst floor falls to 20ms and the same evidence confirms.
        BargeInDecision relaxed = accumulator.AggregateRecent(
            WindowMs,
            requiredDurationMs: 60,
            minRms: 0f,
            BargeInDecisionParameters.Default);

        Assert.Equal(BargeInAction.Interrupt, relaxed.Action);
    }

    [Fact]
    public void EchoFractionLimit_StillApplies()
    {
        // A window dominated by causal echo evidence must not confirm even when a long
        // accepted run is present at the end.
        var frames = new List<NearEndFrameDecision>();
        for (int i = 0; i < 9; i++)
            frames.Add(Frame(false, echoLike: true, rejectDecay: 1.0));
        for (int i = 0; i < 9; i++)
            frames.Add(Frame(true));

        BargeInDecision decision = Aggregate(frames.ToArray());

        Assert.Equal(BargeInAction.Continue, decision.Action);
        Assert.Equal("too-much-causal-echo-evidence", decision.Reason);
    }
}
