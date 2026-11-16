namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct NearEndFrameDecision(
        long TimestampTicks,
        double DurationMs,
        bool Accepted,
        bool EchoLike,
        bool RenderActive,
        double CaptureRms,
        double RequiredRms,
        double RejectDecay,
        string Reason,
        long StreamEpoch = 0,
        long StartSample = 0,
        long EndSampleExclusive = 0,
        bool HasDiscontinuity = false,
        bool ProvisionalPlaybackReferenceGap = false,
        double PostAecResidualRatio = double.NaN,
        double RenderCorrelation = double.NaN,
        double PostAecRenderCorrelation = double.NaN);
