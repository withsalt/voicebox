using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Realtime.Timeline;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RenderAlignmentStageTests
{
    [Fact]
    public void IsFreshDelayHint_RejectsMissingInvalidAndStaleHints()
    {
        var time = new TickTimeProvider();
        var stage = new RenderAlignmentStage(time, delayHintMaxAgeMs: 500, refinedEstimateHoldoverSamples: 16000);

        Assert.False(stage.IsFreshDelayHint(null, 10_000));
        Assert.False(stage.IsFreshDelayHint(new RenderDelayHint(-1, 1, IsStable: true), 10_000));
        Assert.False(stage.IsFreshDelayHint(new RenderDelayHint(20, 0, IsStable: true), 10_000));
        Assert.False(stage.IsFreshDelayHint(new RenderDelayHint(20, 10_000, IsStable: true), 0));

        long start = TimeSpan.FromMilliseconds(100).Ticks;
        long stale = start + TimeSpan.FromMilliseconds(501).Ticks;
        Assert.False(stage.IsFreshDelayHint(new RenderDelayHint(20, start, IsStable: true), stale));

        long fresh = start + TimeSpan.FromMilliseconds(500).Ticks;
        Assert.True(stage.IsFreshDelayHint(new RenderDelayHint(20, start, IsStable: true), fresh));
    }

    [Fact]
    public void HasRecentRefinedEstimate_UsesInclusiveHoldoverWindow()
    {
        var stage = new RenderAlignmentStage(
            new TickTimeProvider(),
            delayHintMaxAgeMs: 500,
            refinedEstimateHoldoverSamples: 100);

        Assert.False(stage.HasRecentRefinedEstimate(0));
        stage.LastRefinedEstimateSample = 50;
        Assert.True(stage.HasRecentRefinedEstimate(50));
        Assert.True(stage.HasRecentRefinedEstimate(150));
        Assert.False(stage.HasRecentRefinedEstimate(151));
        Assert.False(stage.HasRecentRefinedEstimate(49));
        stage.Reset();
        Assert.False(stage.HasRecentRefinedEstimate(50));
    }

    [Fact]
    public void SelectDelayEstimate_FollowsRefinedThenFallbackThenHoldoverThenHints()
    {
        var stage = new RenderAlignmentStage(
            new TickTimeProvider(),
            delayHintMaxAgeMs: 500,
            refinedEstimateHoldoverSamples: 1000);

        var refined = new RenderDelayEstimate(true, 32, 0.9, IsRefined: true);
        Assert.Equal(refined, stage.SelectDelayEstimate(in refined, 10));

        var fallback = new RenderDelayEstimate(true, 48, 0.4, IsCorrelationFallback: true);
        Assert.Equal(fallback, stage.SelectDelayEstimate(in fallback, 10));

        stage.LastRefinedEstimateSample = 0;
        var empty = default(RenderDelayEstimate);
        Assert.Equal(empty, stage.SelectDelayEstimate(in empty, 10));

        stage.Reset();
        stage.DelayHint = new RenderDelayHint(10, 1, IsStable: true);
        RenderDelayEstimate stable = stage.SelectDelayEstimate(in empty, 10);
        Assert.True(stable.HasEstimate);
        Assert.True(stable.IsRefined);
        Assert.True(stable.IsExternalHint);
        Assert.Equal(160.0, stable.DelaySamples);

        stage.DelayHint = null;
        var local = new RenderDelayEstimate(true, 80, 0.2);
        Assert.Equal(local, stage.SelectDelayEstimate(in local, 10));

        stage.DelayHint = new RenderDelayHint(12, 1, IsStable: false);
        var none = default(RenderDelayEstimate);
        RenderDelayEstimate unstable = stage.SelectDelayEstimate(in none, 10);
        Assert.True(unstable.HasEstimate);
        Assert.False(unstable.IsRefined);
        Assert.True(unstable.IsExternalHint);
        Assert.Equal(192.0, unstable.DelaySamples);
    }

    private sealed class TickTimeProvider : TimeProvider
    {
        public override long TimestampFrequency => TimeSpan.TicksPerSecond;
    }
}
