using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies
{
    /// <summary>
    /// Allocation-free time-window accumulator that converts frame evidence into the
    /// module's only public behavior: continue playback or interrupt playback.
    /// </summary>
    internal sealed class BargeInDecisionAccumulator
    {
        private const double ConfirmationTailMs = 80;
        private const double ConfirmationTailMinAcceptedFraction = 0.50;
        private const double ModelEvidenceMinRmsFraction = 0.50;

        private readonly NearEndFrameDecision[] _frames;
        private int _writeIndex;

        public BargeInDecisionAccumulator(int capacity)
        {
            _frames = new NearEndFrameDecision[Math.Max(1, capacity)];
        }

        public int Count { get; private set; }

        public void Reset()
        {
            Array.Clear(_frames);
            _writeIndex = 0;
            Count = 0;
        }

        public void Add(NearEndFrameDecision frame)
        {
            _frames[_writeIndex] = frame;
            _writeIndex = (_writeIndex + 1) % _frames.Length;
            if (Count < _frames.Length)
                Count++;
        }

        public BargeInDecision AggregateRecent(
            double windowMs,
            double requiredDurationMs,
            float minRms,
            BargeInDecisionParameters parameters)
        {
            int length = _frames.Length;
            int scanned = 0;
            int included = 0;
            double selectedDurationMs = 0;
            if (windowMs > 0)
            {
                double accumulatedMs = 0;
                for (int i = 0; i < Count; i++)
                {
                    int index = (_writeIndex - 1 - i + length) % length;
                    scanned = i + 1;
                    double frameDurationMs = _frames[index].DurationMs;
                    if (frameDurationMs <= 0)
                        continue;

                    accumulatedMs += frameDurationMs;
                    included++;
                    if (accumulatedMs >= windowMs)
                        break;
                }

                selectedDurationMs = accumulatedMs;
            }

            if (included == 0)
            {
                return new BargeInDecision(
                    BargeInAction.Continue,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    requiredDurationMs,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    "no-recent-frames");
            }

            int acceptedFrames = 0;
            int rejectedFrames = 0;
            int echoFrames = 0;
            double totalMs = 0;
            double acceptedMs = 0;
            double echoMs = 0;
            double scoreMs = 0;
            double peakScoreMs = 0;
            double lastRms = 0;
            double lastRequired = 0;
            double minRequired = double.PositiveInfinity;
            double maxRequired = 0;
            int postAecEvidenceFrameCount = 0;
            double minPostAecResidualRatio = double.PositiveInfinity;
            double maxPostAecResidualRatio = 0;
            double postAecResidualRatioSum = 0;
            int renderCorrelationCount = 0;
            double renderCorrelationSum = 0;
            int postAecRenderCorrelationCount = 0;
            double postAecRenderCorrelationSum = 0;
            double tailDurationMs = 0;
            double tailAcceptedMs = 0;
            double contiguousAcceptedMs = 0;
            double longestContiguousAcceptedMs = 0;
            bool latestFrameAccepted = false;
            long streamEpoch = 0;
            long startSample = 0;
            long endSample = 0;
            bool hasTimeline = false;
            bool hasDiscontinuity = false;
            long firstAcceptedStartSample = -1;
            bool hasAcceptedTimeline = false;
            string reason = "none";

            // Replay selected frames from oldest to newest so duration scoring and
            // capture-timeline validation have deterministic order.
            for (int i = scanned - 1; i >= 0; i--)
            {
                int index = (_writeIndex - 1 - i + length) % length;
                NearEndFrameDecision frame = _frames[index];
                double durationMs = frame.DurationMs;
                if (durationMs <= 0)
                    continue;

                if (frame.StreamEpoch > 0)
                {
                    if (!hasTimeline)
                    {
                        hasTimeline = true;
                        streamEpoch = frame.StreamEpoch;
                        startSample = frame.StartSample;
                    }
                    else if (streamEpoch != frame.StreamEpoch || endSample != frame.StartSample)
                    {
                        hasDiscontinuity = true;
                    }

                    endSample = frame.EndSampleExclusive;
                    hasDiscontinuity |= frame.HasDiscontinuity;
                }

                bool accepted =
                    frame.Accepted &&
                    !frame.ProvisionalPlaybackReferenceGap;
                double rejectDecay = frame.RejectDecay;
                string frameReason = frame.ProvisionalPlaybackReferenceGap
                    ? "transient-playback-reference-gap"
                    : frame.Reason;
                double effectiveMinRms = frame.RenderActive
                    ? minRms * ModelEvidenceMinRmsFraction
                    : minRms;
                if (effectiveMinRms > 0 && accepted && frame.CaptureRms < effectiveMinRms)
                {
                    accepted = false;
                    rejectDecay = Math.Max(rejectDecay, 0.22);
                    frameReason = frame.RenderActive
                        ? "below-model-backed-min-rms"
                        : "below-min-rms";
                }

                double frameStartMs = totalMs;
                totalMs += durationMs;
                lastRms = frame.CaptureRms;
                lastRequired = frame.RequiredRms;
                minRequired = Math.Min(minRequired, frame.RequiredRms);
                maxRequired = Math.Max(maxRequired, frame.RequiredRms);

                if (accepted)
                {
                    acceptedFrames++;
                    acceptedMs += durationMs;
                    scoreMs += durationMs;
                    contiguousAcceptedMs += durationMs;
                    longestContiguousAcceptedMs = Math.Max(
                        longestContiguousAcceptedMs,
                        contiguousAcceptedMs);
                    if (!hasAcceptedTimeline && frame.StreamEpoch > 0 && frame.StartSample >= 0)
                    {
                        firstAcceptedStartSample = frame.StartSample;
                        hasAcceptedTimeline = true;
                    }
                }
                else
                {
                    rejectedFrames++;
                    contiguousAcceptedMs = 0;
                    scoreMs = Math.Max(0, scoreMs - durationMs * rejectDecay);
                    reason = frameReason;
                }

                double tailStartMs = Math.Max(0, selectedDurationMs - ConfirmationTailMs);
                double tailOverlapMs = Math.Max(
                    0,
                    Math.Min(totalMs, selectedDurationMs) - Math.Max(frameStartMs, tailStartMs));
                tailDurationMs += tailOverlapMs;
                if (accepted)
                    tailAcceptedMs += tailOverlapMs;
                latestFrameAccepted = accepted;

                if (frame.EchoLike)
                {
                    echoFrames++;
                    echoMs += durationMs;
                }

                if (double.IsFinite(frame.PostAecResidualRatio) && frame.PostAecResidualRatio >= 0)
                {
                    postAecEvidenceFrameCount++;
                    minPostAecResidualRatio = Math.Min(
                        minPostAecResidualRatio,
                        frame.PostAecResidualRatio);
                    maxPostAecResidualRatio = Math.Max(
                        maxPostAecResidualRatio,
                        frame.PostAecResidualRatio);
                    postAecResidualRatioSum += frame.PostAecResidualRatio;
                }

                if (double.IsFinite(frame.RenderCorrelation))
                {
                    renderCorrelationCount++;
                    renderCorrelationSum += Math.Abs(frame.RenderCorrelation);
                }

                if (double.IsFinite(frame.PostAecRenderCorrelation))
                {
                    postAecRenderCorrelationCount++;
                    postAecRenderCorrelationSum += Math.Abs(frame.PostAecRenderCorrelation);
                }

                peakScoreMs = Math.Max(peakScoreMs, scoreMs);
            }

            // Require a continuous speech-like run as well as accumulated evidence; isolated
            // near-end spikes do not qualify.
            bool hasEnoughNearEnd =
                acceptedMs >= requiredDurationMs * parameters.MinNearendFraction &&
                longestContiguousAcceptedMs >=
                    requiredDurationMs * parameters.MinContiguousNearendFraction;
            bool echoWithinLimit =
                totalMs <= 0 || echoMs <= totalMs * parameters.MaxEchoFraction;
            bool tailStillNearEnd =
                latestFrameAccepted &&
                tailDurationMs > 0 &&
                tailAcceptedMs >= tailDurationMs * ConfirmationTailMinAcceptedFraction;
            bool shouldInterrupt =
                scoreMs >= requiredDurationMs &&
                hasEnoughNearEnd &&
                echoWithinLimit &&
                tailStillNearEnd &&
                !hasDiscontinuity;

            if (hasDiscontinuity)
                reason = "capture-timeline-discontinuity";
            else if (shouldInterrupt)
                reason = "confirmed-nearend-window";
            else if (acceptedMs < requiredDurationMs * parameters.MinNearendFraction)
                reason = "insufficient-nearend-duration";
            else if (!hasEnoughNearEnd)
                reason = "insufficient-contiguous-nearend-evidence";
            else if (!echoWithinLimit)
                reason = "too-much-causal-echo-evidence";
            else if (scoreMs < requiredDurationMs)
                reason = "insufficient-score-duration";
            else if (!tailStillNearEnd)
                reason = "insufficient-recent-nearend-evidence";

            return new BargeInDecision(
                shouldInterrupt ? BargeInAction.Interrupt : BargeInAction.Continue,
                included,
                acceptedFrames,
                rejectedFrames,
                echoFrames,
                totalMs,
                acceptedMs,
                requiredDurationMs,
                scoreMs,
                peakScoreMs,
                lastRms,
                lastRequired,
                double.IsPositiveInfinity(minRequired) ? 0 : minRequired,
                maxRequired,
                reason,
                streamEpoch,
                startSample,
                endSample,
                hasDiscontinuity,
                latestFrameAccepted,
                postAecEvidenceFrameCount,
                double.IsPositiveInfinity(minPostAecResidualRatio) ? double.PositiveInfinity : minPostAecResidualRatio,
                maxPostAecResidualRatio,
                postAecEvidenceFrameCount > 0
                    ? postAecResidualRatioSum / postAecEvidenceFrameCount
                    : double.NaN,
                renderCorrelationCount > 0
                    ? renderCorrelationSum / renderCorrelationCount
                    : double.NaN,
                postAecRenderCorrelationCount > 0
                    ? postAecRenderCorrelationSum / postAecRenderCorrelationCount
                    : double.NaN,
                firstAcceptedStartSample);
        }
    }
}
