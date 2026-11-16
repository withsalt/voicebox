namespace VoiceBox.Assistant.Modules.Speaker.Configuration;

public sealed class SpeakerEnrollmentOptions
{
    /// <summary>Clean segments to record. Default 5.</summary>
    public int CleanSegmentCount { get; set; } = 5;

    /// <summary>Double-talk segments to record; 0 skips the doubleTalkNearEnd bank. Default 3.</summary>
    public int DoubleTalkSegmentCount { get; set; } = 3;

    /// <summary>Owner speech per segment. Default 2.5s.</summary>
    public TimeSpan SegmentSpeech { get; set; } = TimeSpan.FromSeconds(2.5);

    /// <summary>Wall-clock budget for one segment attempt. Default 12s.</summary>
    public TimeSpan SegmentTimeout { get; set; } = TimeSpan.FromSeconds(12);

    /// <summary>Owner speech per independent confirmation segment. Default 2.5s.</summary>
    public TimeSpan ConfirmationSpeech { get; set; } = TimeSpan.FromSeconds(2.5);

    public int MaxAttemptsPerSegment { get; set; } = 3;

    public int MinCleanSegmentsRetained { get; set; } = 4;

    public int MinDoubleTalkSegmentsRetained { get; set; } = 2;

    /// <summary>Minimum separate retained segments to select before suppressing duplicate directions, when available. Default 2.</summary>
    public int MinPrototypesPerBank { get; set; } = 2;

    /// <summary>Maximum representative prototypes; duplicate directions may leave slots unused. Default 4.</summary>
    public int MaxPrototypesPerBank { get; set; } = 4;

    public float MinimumRms { get; set; } = 0.005f;

    public double MaxClippingRatio { get; set; } = 0.002;

    public double MaxDcOffset { get; set; } = 0.02;

    public double MinSpeechRatio { get; set; } = 0.55;

    public double MaxSpeechRatio { get; set; } = 0.98;

    /// <summary>Double-talk segments must have render active for at least this share of their frames. Default 0.7.</summary>
    public double MinRenderActiveRatio { get; set; } = 0.7;

    /// <summary>Double-talk segments must have DTD-accepted near-end frames for at least this share. Default 0.6.</summary>
    public double MinNearEndAcceptedRatio { get; set; } = 0.6;

    /// <summary>Minimum median cosine similarity between a segment's own windows. Default 0.3.</summary>
    public double MinWithinSegmentSimilarity { get; set; } = 0.3;

    /// <summary>Outlier rule: <c>score &lt; median - OutlierMadMultiplier * max(MAD, OutlierMadFloor)</c>.</summary>
    public double OutlierMadMultiplier { get; set; } = 3.0;

    public double OutlierMadFloor { get; set; } = 0.02;

    /// <summary>Render RMS above which a duplex frame counts as "render active" during double-talk.</summary>
    public float RenderActiveRms { get; set; } = 0.01f;
}
