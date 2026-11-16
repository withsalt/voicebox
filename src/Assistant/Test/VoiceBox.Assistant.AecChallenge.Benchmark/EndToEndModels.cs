using VoiceBox.Core.Onnx;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed record EndToEndOptions
{
    public required string DatasetRoot { get; init; }
    public required string AsrModelPath { get; init; }
    public required string TtsModelPath { get; init; }
    public string? TtsVoice { get; init; }
    public string? VadModelDirectory { get; init; }
    public OnnxExecutionProvider ExecutionProvider { get; init; } = OnnxExecutionProvider.Cuda;
    public OnnxExecutionProvider TtsExecutionProvider => OnnxExecutionProvider.Cpu;
    public string Suite { get; init; } = "release-e2e";
    public int? Limit { get; init; }
    public string? CaseId { get; init; }
    public string? OutputPath { get; init; }
    public int AsrTimeoutSeconds { get; init; } = 120;
    public bool Verbose { get; init; }
    public bool EnforceGates { get; init; }
}

internal sealed record EndToEndCaseResult
{
    public string Id { get; init; } = string.Empty;
    public string ClusterId { get; init; } = string.Empty;
    public string Corpus { get; init; } = string.Empty;
    public string? SpeechAssetId { get; init; }
    public bool ExpectedNearEnd { get; init; }
    public string Scenario { get; init; } = string.Empty;
    public string GeometryId { get; init; } = string.Empty;
    public string SpeakerPolicy { get; init; } = "any-speaker";
    public bool ExpectHeard { get; init; }
    public bool ExpectDuck { get; init; }
    public bool ExpectAccepted { get; init; }
    public bool ExpectPlaybackStop { get; init; }
    public bool ExpectPlaybackResume { get; init; }
    public string ReferenceText { get; init; } = string.Empty;
    public string RecognizedText { get; init; } = string.Empty;
    public bool Heard { get; init; }
    public bool EarlyFalseTrigger { get; init; }
    public bool Accepted { get; init; }
    public bool Rejected { get; init; }
    public bool PlaybackDucked { get; init; }
    public bool PlaybackStopped { get; init; }
    public bool PlaybackResumed { get; init; }
    public string? ValidationReason { get; init; }
    public int CallbackSamples { get; init; }
    public double AudioDurationMs { get; init; }
    /// <summary>
    /// Reaction latency from the earliest sample at which a correct implementation is allowed to
    /// fire (near-end speech present, minimum duration available, startup guard elapsed).
    /// </summary>
    public double? HeardLatencyMs { get; init; }

    /// <summary>
    /// Reaction latency from the first sample of near-end speech energy. Differs from
    /// <see cref="HeardLatencyMs"/> by the product's own minimum-duration and startup-guard budget,
    /// and is reported so the two contributions can be separated.
    /// </summary>
    public double? HeardLatencyFromSpeechOnsetMs { get; init; }

    /// <summary>Control-plane duck request latency.</summary>
    public double? DuckLatencyMs { get; init; }

    /// <summary>
    /// Latency until the loudspeaker actually emitted attenuated audio. Exceeds
    /// <see cref="DuckLatencyMs"/> by the committed device output latency plus the gain ramp, and is
    /// the figure that corresponds to perceived responsiveness.
    /// </summary>
    public double? AudibleDuckLatencyMs { get; init; }

    public double? StopLatencyMs { get; init; }
    public double? AudibleStopLatencyMs { get; init; }
    public double? AudibleTailLatencyMs { get; init; }

    /// <summary>Latency until the committed device buffer finished draining, excluding room reverberation.</summary>
    public double? DrainCompleteLatencyMs { get; init; }

    /// <summary>
    /// Portion of the stop tail that no implementation can remove, because the device had already
    /// accepted that audio when the stop was issued.
    /// </summary>
    public double? DeviceTailLatencyMs { get; init; }

    /// <summary>Where the near-end clip was spliced in. Scheduling position only, kept for continuity.</summary>
    public double NearEndInsertionOnsetMs { get; init; }

    /// <summary>First near-end speech energy in the microphone signal.</summary>
    public double? NearEndSpeechOnsetMs { get; init; }

    /// <summary>Earliest sample at which a barge-in report is legal.</summary>
    public double? NearEndEligibleOnsetMs { get; init; }

    /// <summary>
    /// False when the near-end utterance never offers a qualifying speech run outside the startup
    /// guard. Such a case cannot be expected to trigger and must not be scored as a miss.
    /// </summary>
    public bool NearEndEligible { get; init; }

    public double NearEndActiveDurationMs { get; init; }
    public string? TriggerSource { get; init; }
    public int DecisionFrameCount { get; init; }
    public int DecisionAcceptedFrameCount { get; init; }
    public int DecisionEchoLikeFrameCount { get; init; }
    public double DecisionAcceptedDurationMs { get; init; }
    public double DecisionScoreDurationMs { get; init; }
    public double DecisionPeakScoreDurationMs { get; init; }
    public int CharacterErrors { get; init; }
    public int ReferenceCharacters { get; init; }
    public double? CharacterErrorRate { get; init; }
    public bool ExactMatch { get; init; }
    public bool EmptyRecognition { get; init; }
    public double SignalToEchoRatioDb { get; init; }
    public double SignalToNoiseRatioDb { get; init; }

    /// <summary>Realized near-end active speech level. Fixed by design; reported to prove it is fixed.</summary>
    public double NearEndActiveLevel { get; init; }

    /// <summary>Realized echo active level. This is the quantity SER moves.</summary>
    public double EchoActiveLevel { get; init; }

    /// <summary>
    /// SER actually produced by the scene, from the realized levels. Should track
    /// <see cref="SignalToEchoRatioDb"/>; a divergence means the level model failed for this case.
    /// </summary>
    public double? RealizedSignalToEchoRatioDb { get; init; }

    public double EchoReturnLossDb { get; init; }
    public double SpeakerGain { get; init; }

    /// <summary>Fraction of microphone samples that hit the clip bound.</summary>
    public double ClippedSampleRatio { get; init; }

    public int EchoDelayMs { get; init; }
    public int ClockDriftPpm { get; init; }
    public bool FarEndNonlinear { get; init; }
    public int? RenderGapStartMs { get; init; }
    public int RenderGapDurationMs { get; init; }
    public string EchoRirId { get; init; } = string.Empty;
    public string? NearRirId { get; init; }
    public string? NoiseAssetId { get; init; }
    public string? NoiseKind { get; init; }
    public string? NoiseCorpus { get; init; }
    public string? Error { get; init; }
}

internal sealed record EndToEndSummary
{
    public int Total { get; init; }
    public int ClusterCount { get; init; }
    public int Positive { get; init; }
    public int Negative { get; init; }
    public int ErrorCount { get; init; }
    public int HeardPositive { get; init; }
    public int AcceptedPositive { get; init; }
    public int DuckedPositive { get; init; }
    public int StoppedPositive { get; init; }
    public int ResumedPositive { get; init; }
    public int RejectedPositive { get; init; }
    public int EarlyFalseTrigger { get; init; }
    public int EchoOnlyHeard { get; init; }
    public int EchoOnlyAccepted { get; init; }
    public double? HeardRecall { get; init; }
    public double? AcceptedRecall { get; init; }
    public double? DuckRecall { get; init; }
    public double? PlaybackStopRecall { get; init; }
    public double? PlaybackResumeRecall { get; init; }
    public double? AggregateCharacterErrorRate { get; init; }
    public double? ExactMatchRate { get; init; }
    public double? EmptyRecognitionRate { get; init; }
    public double? HeardLatencyP50Ms { get; init; }
    public double? HeardLatencyP95Ms { get; init; }
    public double? StopLatencyP95Ms { get; init; }
    public double? AudibleStopLatencyP95Ms { get; init; }
    public double? AudibleTailLatencyP95Ms { get; init; }
    public int HeardLatencyCensored { get; init; }
    public int StopLatencyCensored { get; init; }
    public int AudibleStopLatencyCensored { get; init; }
    public int AudibleTailLatencyCensored { get; init; }

    /// <summary>Control-plane duck latency percentile.</summary>
    public double? DuckLatencyP95Ms { get; init; }

    /// <summary>
    /// Latency until the loudspeaker actually attenuated. Ducking is the only arm of the barge-in
    /// sequence that runs in real time, so this is the responsiveness figure.
    /// </summary>
    public double? AudibleDuckLatencyP95Ms { get; init; }

    /// <summary>Latency until the committed device buffer finished draining, excluding reverberation.</summary>
    public double? DrainCompleteLatencyP95Ms { get; init; }

    /// <summary>Irreducible device-side stop tail.</summary>
    public double? DeviceTailLatencyP95Ms { get; init; }

    /// <summary>
    /// Share of accepted barge-ins whose stop was authorized by the candidate's own speaker Match
    /// (KWS enabled, reason <c>speaker-match</c>). Recognition text never carries the decision.
    /// The property name is kept for report/JSON compatibility.
    /// </summary>
    public double? TextValidatedAcceptRate { get; init; }

    /// <summary>
    /// Share of accepted barge-ins whose stop was authorized by the acoustic Interrupt alone (KWS
    /// disabled, reason <c>audio-confirmed</c>). The property name is kept for report/JSON compatibility.
    /// </summary>
    public double? TimeoutAcceptRate { get; init; }

    /// <summary>
    /// False accepts per hour of echo-only audio. Replaces a raw count so the figure does not move
    /// with dataset size. These are adversarial hours: continuous echo, not typical usage.
    /// </summary>
    public double? FalseAcceptPerHour { get; init; }

    public double NegativeAudioHours { get; init; }
}

internal sealed record EndToEndReport
{
    public required string SchemaVersion { get; init; }
    public required DateTimeOffset CreatedUtc { get; init; }
    public required string DatasetRoot { get; init; }
    public required string DatasetId { get; init; }
    public required string AsrModelPath { get; init; }
    public required string TtsModelPath { get; init; }
    public required string ExecutionProvider { get; init; }
    public required string TtsExecutionProvider { get; init; }
    public required double WallTimeSeconds { get; init; }
    public required double AudioTimeSeconds { get; init; }
    public required double RealTimeFactor { get; init; }
    public required EndToEndSummary Overall { get; init; }
    public required IReadOnlyList<EndToEndSummarySlice> Slices { get; init; }
    public required IReadOnlyList<QualityGate> Gates { get; init; }
    public required IReadOnlyList<EndToEndCaseResult> Cases { get; init; }
}

internal sealed record EndToEndSummarySlice
{
    public required string Name { get; init; }
    public required EndToEndSummary Summary { get; init; }
}
