using VoiceBox.Core.Onnx;

namespace VoiceBox.AEC.DTLN.Models
{
    /// <summary>
    /// Configuration for <see cref="DtlnAecService"/>.
    /// DTLN-aec pretrained models are 16 kHz mono streaming models.
    /// </summary>
    public sealed class DtlnAecOptions : OnnxExecutionOptions
    {
        public const int SupportedSampleRate = 16000;
        public const int SupportedChannels = 1;
        public const int BlockLength = 512;
        public const int HopLength = 128;
        public const int FrequencyBins = BlockLength / 2 + 1;

        private string? _modelPart1Path;
        private string? _modelPart2Path;

        /// <summary>
        /// ONNX Runtime intra-op threads. DTLN-aec runs every 8 ms, so a small fixed thread count
        /// avoids real-time callback scheduling noise.
        /// </summary>
        public override int ThreadsNum { get; set; } = 1;

        /// <summary>Input sample rate. Only 16 kHz is supported.</summary>
        public uint SampleRate { get; set; } = SupportedSampleRate;

        /// <summary>Input channel count. Only mono is supported.</summary>
        public uint ChannelCount { get; set; } = SupportedChannels;

        /// <summary>Pretrained model size to load when explicit model paths are not provided.</summary>
        public DtlnAecModelSize ModelSize { get; set; } = DtlnAecModelSize.Units128;

        /// <summary>Path to the stage-1 ONNX model. Null loads the bundled model for <see cref="ModelSize"/>.</summary>
        public string? ModelPart1Path
        {
            get => _modelPart1Path;
            set => _modelPart1Path = string.IsNullOrWhiteSpace(value) ? null : value;
        }

        /// <summary>Path to the stage-2 ONNX model. Null loads the bundled model for <see cref="ModelSize"/>.</summary>
        public string? ModelPart2Path
        {
            get => _modelPart2Path;
            set => _modelPart2Path = string.IsNullOrWhiteSpace(value) ? null : value;
        }

        /// <summary>
        /// Delay render reference by this many samples before feeding DTLN-aec. Use this when the
        /// physical speaker-to-microphone echo arrives later than the callback-aligned render span.
        /// <para>
        /// When <see cref="EnableAutoDelayEstimation"/> is true this is only the initial seed: the
        /// WebRTC AEC3 estimator takes over and tracks the real round-trip delay automatically.
        /// When auto estimation is off, this is the fixed delay used for the whole session.
        /// </para>
        /// </summary>
        public int ReferenceDelaySamples { get; set; }

        /// <summary>
        /// Enable online delay estimation. When true, the speaker-to-microphone round-trip
        /// delay is measured continuously from the render/capture signals and applied automatically,
        /// so <see cref="ReferenceDelaySamples"/> does not need to be tuned per device. Default true.
        /// </summary>
        public bool EnableAutoDelayEstimation { get; set; } = true;

        /// <summary>
        /// Maximum round-trip delay (ms) the estimator searches for. Must cover the worst-case device
        /// latency (Bluetooth output can reach ~200 ms). Default 300.
        /// Only used when <see cref="EnableAutoDelayEstimation"/> is true.
        /// </summary>
        public int MaxAutoDelayMs { get; set; } = 300;

        /// <summary>Maximum absolute audio sample after processing.</summary>
        public float ClipLimit { get; set; } = 1.0f;

        /// <summary>
        /// Number of zero hops processed during reset/start. The reference TFLite script effectively
        /// has three zero hops before the first real output because it pads by block_len-hop.
        /// </summary>
        public int WarmupHops { get; set; } = 3;

        /// <summary>
        /// Prime output with this many zeros so callback sizes that are not aligned to the
        /// 128-sample DTLN hop can be served without initial underflow.
        /// </summary>
        public int OutputPrerollSamples { get; set; } = HopLength;

        internal void Validate()
        {
            if (SampleRate != SupportedSampleRate)
                throw new NotSupportedException($"DTLN AEC supports {SupportedSampleRate} Hz only, got {SampleRate}.");
            if (ChannelCount != SupportedChannels)
                throw new NotSupportedException($"DTLN AEC supports mono audio only, got {ChannelCount} channels.");
            _ = ModelSize.Units();
            if (ReferenceDelaySamples < 0)
                throw new ArgumentOutOfRangeException(nameof(ReferenceDelaySamples), ReferenceDelaySamples, "ReferenceDelaySamples must be >= 0.");
            if (EnableAutoDelayEstimation)
            {
                if (MaxAutoDelayMs <= 0)
                    throw new ArgumentOutOfRangeException(nameof(MaxAutoDelayMs), MaxAutoDelayMs, "MaxAutoDelayMs must be > 0.");
                int maxAutoDelaySamples = MaxAutoDelayMs * (int)SampleRate / 1000;
                if (ReferenceDelaySamples > maxAutoDelaySamples)
                    throw new ArgumentOutOfRangeException(nameof(ReferenceDelaySamples), ReferenceDelaySamples,
                        $"ReferenceDelaySamples seed ({ReferenceDelaySamples}) must be <= MaxAutoDelayMs in samples ({maxAutoDelaySamples}).");
            }
            if (ClipLimit <= 0 || ClipLimit > 1 || float.IsNaN(ClipLimit))
                throw new ArgumentOutOfRangeException(nameof(ClipLimit), ClipLimit, "ClipLimit must be in (0, 1].");
            if (WarmupHops < 0)
                throw new ArgumentOutOfRangeException(nameof(WarmupHops), WarmupHops, "WarmupHops must be >= 0.");
            if (OutputPrerollSamples < 0)
                throw new ArgumentOutOfRangeException(nameof(OutputPrerollSamples), OutputPrerollSamples, "OutputPrerollSamples must be >= 0.");
            if ((ModelPart1Path is null) != (ModelPart2Path is null))
                throw new ArgumentException("ModelPart1Path and ModelPart2Path must be provided together.");
        }
    }
}
