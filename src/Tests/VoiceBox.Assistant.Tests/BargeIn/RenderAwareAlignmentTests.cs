using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Assistant.Realtime.Timeline;
using VoiceBox.Core.Services.AudioService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RenderAwareAlignmentTests
{
    [Fact]
    public void FractionalRenderBufferInterpolatesAndRemainsBounded()
    {
        var buffer = new BoundedFractionalRenderBuffer(4);
        buffer.Append([0f, 1f, 2f, 3f], 0);
        buffer.Append(4f, 4);

        Assert.Equal(4, buffer.Count);
        Assert.Equal(1, buffer.StartSample);
        Assert.Equal(5, buffer.EndSampleExclusive);

        float[] output = new float[2];
        Assert.True(buffer.TryRead(2.5, output));
        Assert.Equal([2.5f, 3.5f], output);
        Assert.False(buffer.TryRead(0, output));
    }

    [Fact]
    public void FractionalRenderBufferRejectsTimelineGaps()
    {
        var buffer = new BoundedFractionalRenderBuffer(8);
        buffer.Append([1f, 2f], 10);
        Assert.Throws<InvalidOperationException>(() => buffer.Append([3f], 13));
    }

    [Fact]
    public void FractionalRenderBufferRejectsReadsAcrossInvalidRenderSamples()
    {
        var buffer = new BoundedFractionalRenderBuffer(8);
        buffer.Append([1f, 2f], 0);
        buffer.Append([0f, 0f], 2, isValid: false);
        buffer.Append([5f, 6f], 4);

        float[] output = new float[2];
        Assert.True(buffer.TryRead(0, output, out bool firstInvalid));
        Assert.False(firstInvalid);
        Assert.False(buffer.TryRead(1, output, out bool intersectsInvalid));
        Assert.True(intersectsInvalid);
        Assert.True(buffer.TryRead(4, output, out bool lastInvalid));
        Assert.False(lastInvalid);
    }

    [Fact]
    public void TimelineResamplerDropsBufferedSamplesBeforeANewOrigin()
    {
        var resampler = new TimelineMonoResampler();
        var destination = new BoundedFractionalRenderBuffer(1024);
        float[] first = Enumerable.Range(0, 320).Select(value => (float)value).ToArray();
        float[] second = Enumerable.Range(320, 320).Select(value => (float)value).ToArray();
        var firstRange = new CaptureRange(1, 1, 1, 0, 320, 1);
        var secondRange = new CaptureRange(1, 2, 2, 320, 640, 2);
        CaptureSignalTimeline firstTimeline =
            CaptureSignalTimeline.FromWholeBuffer(in firstRange, first.Length);
        CaptureSignalTimeline secondTimeline =
            CaptureSignalTimeline.FromWholeBuffer(in secondRange, second.Length);

        resampler.Append(
            in firstTimeline,
            first,
            sampleRate: 16000,
            channels: 1,
            originSample: 160,
            streamEpoch: 1,
            destination);
        resampler.Append(
            in secondTimeline,
            second,
            sampleRate: 16000,
            channels: 1,
            originSample: 160,
            streamEpoch: 1,
            destination);

        Assert.Equal(0, destination.StartSample);
        float[] output = new float[2];
        Assert.True(destination.TryRead(158, output));
        Assert.Equal([318f, 319f], output);
    }

    [Fact]
    public void DelayTrackerPromotesCoarseEstimateAndTracksDrift()
    {
        var tracker = new AdaptiveRenderDelayTracker(new VoiceAssistantRenderAlignmentOptions
        {
            MinimumAlignmentConfidence = 0.5,
            EstimateHoldoverMs = 1000,
            MaximumDriftPpm = 1000,
        });

        RenderAlignmentUpdate first = tracker.Observe(0, new RenderDelayEstimate(true, 80, 0.8));
        RenderAlignmentUpdate second = tracker.Observe(320, new RenderDelayEstimate(true, 81, 0.8));
        Assert.False(first.IsAligned);
        Assert.False(second.IsAligned);
        RenderAlignmentUpdate promoted = tracker.Observe(640, new RenderDelayEstimate(true, 82, 0.8));
        Assert.True(promoted.IsAligned);
        Assert.InRange(promoted.DelaySamples, 80, 82);

        RenderAlignmentUpdate drifted = tracker.Observe(16000, new RenderDelayEstimate(true, 90, 0.9, IsRefined: true));
        Assert.True(drifted.IsAligned);
        Assert.NotEqual(0, drifted.DriftPpm);
    }

    [Fact]
    public void DelayTrackerResetClearsGenerationStateAfterRenderGap()
    {
        var tracker = new AdaptiveRenderDelayTracker(new VoiceAssistantRenderAlignmentOptions
        {
            MinimumAlignmentConfidence = 0.5,
        });

        tracker.Observe(0, new RenderDelayEstimate(true, 20, 1, IsRefined: true));
        long generation = tracker.Generation;
        tracker.Reset();
        Assert.True(tracker.Generation > generation);
        Assert.False(tracker.Advance(0).IsAligned);
    }

    [Fact]
    public void DelayTrackerCorrectsUnverifiedHintPathWithOneMeasurement()
    {
        var tracker = new AdaptiveRenderDelayTracker(new VoiceAssistantRenderAlignmentOptions
        {
            MinimumAlignmentConfidence = 0.5,
            PathChangeThresholdMs = 40,
            EstimateHoldoverMs = 1000,
        });

        // Three consistent external hints bootstrap an unverified path at 12 ms.
        for (int i = 0; i < 3; i++)
        {
            tracker.Observe(
                i * 320,
                new RenderDelayEstimate(true, 192, 0.65, IsExternalHint: true));
        }
        long generation = tracker.Generation;

        // One stream-local measurement replaces the hint path immediately; the
        // material jump restarts the model because everything inferred so far
        // read the wrong render audio.
        RenderAlignmentUpdate corrected = tracker.Observe(
            960,
            new RenderDelayEstimate(true, 1744, 0.7, IsCorrelationFallback: true));
        Assert.True(corrected.PathChanged);
        Assert.True(corrected.Generation > generation);
        Assert.InRange(corrected.DelaySamples, 1743, 1745);

        // The path is verified now, so one later outlier no longer replaces it.
        RenderAlignmentUpdate outlier = tracker.Observe(
            1280,
            new RenderDelayEstimate(true, 240, 0.9, IsRefined: true));
        Assert.False(outlier.PathChanged);
        Assert.InRange(outlier.DelaySamples, 1743, 1745);
    }

    [Fact]
    public void DelayTrackerSnapsNearbyMeasurementOntoHintPathWithoutReset()
    {
        var tracker = new AdaptiveRenderDelayTracker(new VoiceAssistantRenderAlignmentOptions
        {
            MinimumAlignmentConfidence = 0.5,
            PathChangeThresholdMs = 40,
            EstimateHoldoverMs = 1000,
        });

        for (int i = 0; i < 3; i++)
        {
            tracker.Observe(
                i * 320,
                new RenderDelayEstimate(true, 192, 0.65, IsExternalHint: true));
        }
        long generation = tracker.Generation;

        // Measurement within half a hop of the hint: adopt directly without a
        // model-resetting path change.
        RenderAlignmentUpdate snapped = tracker.Observe(
            960,
            new RenderDelayEstimate(true, 240, 0.7, IsCorrelationFallback: true));
        Assert.False(snapped.PathChanged);
        Assert.Equal(generation, snapped.Generation);
        Assert.InRange(snapped.DelaySamples, 239, 241);

        // Verified by the measurement: a later distant estimate must build
        // path-change consensus instead of replacing the path at once.
        RenderAlignmentUpdate outlier = tracker.Observe(
            1280,
            new RenderDelayEstimate(true, 1600, 0.9, IsRefined: true));
        Assert.False(outlier.PathChanged);
        Assert.Equal(generation, outlier.Generation);
        Assert.InRange(outlier.DelaySamples, 239, 241);
    }

    [Fact]
    public void DelayTrackerRequiresConsistentPathChangeBeforeReset()
    {
        var tracker = new AdaptiveRenderDelayTracker(new VoiceAssistantRenderAlignmentOptions
        {
            MinimumAlignmentConfidence = 0.5,
            PathChangeThresholdMs = 20,
            EstimateHoldoverMs = 1000,
        });

        tracker.Observe(0, new RenderDelayEstimate(true, 80, 1, IsRefined: true));
        long generation = tracker.Generation;

        RenderAlignmentUpdate transient = tracker.Observe(
            320,
            new RenderDelayEstimate(true, 640, 1, IsRefined: true));
        Assert.False(transient.PathChanged);
        Assert.Equal(generation, transient.Generation);

        RenderAlignmentUpdate second = tracker.Observe(
            640,
            new RenderDelayEstimate(true, 642, 1, IsRefined: true));
        Assert.False(second.PathChanged);

        RenderAlignmentUpdate promoted = tracker.Observe(
            960,
            new RenderDelayEstimate(true, 641, 1, IsRefined: true));
        Assert.True(promoted.PathChanged);
        Assert.True(promoted.Generation > generation);
        Assert.InRange(promoted.DelaySamples, 640, 642);
    }
}
